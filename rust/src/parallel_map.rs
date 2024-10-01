// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub struct ParallelMap<I, T>
where
    I: Iterator,
    I::Item: Send,
    T: Send,
{
    now: usize,
    iter: I,
    communication: Vec<ThreadCommunication<I::Item, T>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

/// Send `U`, receive `V`.
struct ThreadCommunication<U, V>
where
    U: Send,
    V: Send,
{
    send: std::sync::mpsc::Sender<Option<U>>,
    receive: std::sync::mpsc::Receiver<Option<V>>,
}

impl<Item: Send, T: Send> ThreadCommunication<Item, T> {
    fn new_pair() -> (ThreadCommunication<Item, T>, ThreadCommunication<T, Item>) {
        let (tx_item, rx_item) = std::sync::mpsc::channel::<Option<Item>>();
        let (tx_t, rx_t) = std::sync::mpsc::channel::<Option<T>>();
        // For the ParallelMap (sending Item, receiving T).
        let par_map = ThreadCommunication { send: tx_item, receive: rx_t };
        // For the thread (receiving Item, sending T).
        let thread = ThreadCommunication { send: tx_t, receive: rx_item };
        (par_map, thread)
    }
}

impl<I: Iterator, T: Send> Iterator for ParallelMap<I, T>
where
    I: Iterator,
    I::Item: Send,
    T: Send,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        // If the original iterator was empty we have nothing to return.
        if self.communication.is_empty() {
            return None;
        }

        // Get answer from the thread number `self.now`.
        let result = self.communication[self.now].receive.recv().unwrap_or_default();

        // Some(task) means more work for the thread, None means the thread should finish.
        let _ = self.communication[self.now].send.send(self.iter.next());

        // Move to the next thread (which should be finishing soonest if all tasks take
        // the same time).
        self.now = (self.now + 1) % self.communication.len();

        // Return the result.
        result
    }
}

impl<I, T> Drop for ParallelMap<I, T>
where
    I: Iterator,
    I::Item: Send,
    T: Send,
{
    fn drop(&mut self) {
        // Send end of communication to all threads.
        for communication in &self.communication {
            let _ = communication.send.send(None);
        }
        self.communication.clear();

        // Join all threads.
        while let Some(handle) = self.handles.pop() {
            let _ = handle.join();
        }
    }
}

/// Map `fun` on `iter` using `threads` threads. This can be used with any Iterator and returns an
/// Iterator.
///
/// Example
/// ```
/// use sedpack_rs::parallel_map;
/// let s: i64 = parallel_map::parallel_map(|x| 2 * x, (1 .. 6).into_iter(), 15).sum();
/// assert_eq!(s, 30);
/// ```
pub fn parallel_map<I, T>(fun: fn(I::Item) -> T, mut iter: I, threads: usize) -> ParallelMap<I, T>
where
    I: Iterator,
    I::Item: Clone + Send + 'static,
    T: Send + 'static,
{
    // At least some workers.
    assert!(threads >= 1);

    // Start the threads and send an item to each of them.
    let mut communication = Vec::new();
    let mut handles = Vec::new();
    for t in 0 .. threads {
        // Next task for the thread.
        let next_task = match iter.next() {
            None => break, // Not creating a thread for nothing.
            Some(next_task) => Some(next_task),
        };

        // Create the thread.
        let (par_map, thread) = ThreadCommunication::<I::Item, T>::new_pair();
        communication.push(par_map);
        let handle = std::thread::spawn(move || {
            while let Ok(Some(task)) = thread.receive.recv() {
                match thread.send.send(Some((fun)(task))) {
                    Ok(()) => (),
                    Err(_) => return,
                }
            }
        });
        handles.push(handle);

        // Send the task.
        let _ = communication[t].send.send(next_task);
    }

    ParallelMap { now: 0, iter, communication, handles }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_one(x: i32) -> i32 {
        x + 1
    }

    #[test]
    fn plus_one() {
        for iterator_length in 0 .. 50 {
            for threads in 1 .. 20 {
                let mut iterations = 0;
                for (i, res) in parallel_map(add_one, 0 .. iterator_length, threads).enumerate() {
                    // The value is correct and deterministic.
                    assert_eq!((i + 1) as i32, res);
                    iterations += 1;
                }
                // The resulting iterator is neither shorter or longer.
                assert_eq!(iterations, iterator_length);
            }
        }
    }

    fn sleepy_add_one(x: i32) -> i32 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        x + 1
    }

    #[test]
    fn sleepy_plus_one() {
        for iterator_length in 0 .. 10 {
            for threads in 1 .. 10 {
                let mut iterations = 0;
                let now = std::time::Instant::now();
                for (i, res) in
                    parallel_map(sleepy_add_one, 0 .. iterator_length, threads).enumerate()
                {
                    // The value is correct and deterministic.
                    assert_eq!((i + 1) as i32, res);
                    iterations += 1;
                }
                let elapsed = now.elapsed();
                let expected = std::time::Duration::from_millis(
                    11u64 + 10u64 * (iterator_length as u64 + 1u64) / threads as u64,
                );
                assert!(elapsed < expected);
                // The resulting iterator is neither shorter or longer.
                assert_eq!(iterations, iterator_length);
            }
        }
    }
}
