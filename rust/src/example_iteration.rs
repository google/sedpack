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
use std::io::Read;

use yoke::Yoke;

pub use super::parallel_map::{parallel_map, ParallelMap};
pub use super::shard_generated::sedpack::io::flatbuffer::shardfile::{root_as_shard, Shard};

pub type Example = Vec<Vec<u8>>;
type LoadedShard = Yoke<Shard<'static>, Vec<u8>>;

/// Iterate all examples in given shard files.
pub struct ExampleIterator {
    example_iterator:
        std::iter::Flatten<ParallelMap<<Vec<String> as IntoIterator>::IntoIter, ShardProgress>>,
}

impl ExampleIterator {
    pub fn new(files: Vec<String>, repeat: bool, threads: usize) -> Self {
        assert!(!repeat, "Not implemented yet: repeat=true");
        let example_iterator =
            parallel_map(get_shard_progress, files.into_iter(), threads).flatten();
        ExampleIterator { example_iterator }
    }
}

impl Iterator for ExampleIterator {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        self.example_iterator.next()
    }
}

/// Iterator over a single shard file.
struct ShardProgress {
    total_examples: usize,
    used_examples: usize,
    shard: LoadedShard,
}

/// Get ShardProgress.
fn get_shard_progress(file_path: &str) -> ShardProgress {
    // TODO compressed file support.
    let mut file = std::fs::File::open(file_path).unwrap();
    let mut file_bytes = Vec::new();
    let _ = file.read_to_end(&mut file_bytes).unwrap();

    // A shard is a vector of examples (positive number -- invariant kept by Python code).
    // An example is vector of attributes (the same number of attributes in each example of each
    // shard).
    // An attribute is a vector of bytes.
    // For more details see the FlatBuffers schema.
    //
    // If parsing fails at any time it fails.
    let shard: LoadedShard = Yoke::attach_to_cart(file_bytes, |x| root_as_shard(x).unwrap());
    // Number of examples might be different in different shards.
    let total_examples = shard.get().examples().unwrap().len();

    ShardProgress { total_examples, used_examples: 0, shard }
}

/// Get single example out of a ShardProgress.
///
/// # Arguments
///   
/// * `id` - Id of the example to be returned. Must be in the interval
///   `[shard_progress.used_examples, shard_progress.total_examples)`.
///
/// * `shard_progress` - The shard file information to be used. A copy from this memory happens.
///   Also the `shard_progress.used_examples` is not modified to allow multiple threads to access.
///
/// # Examples
fn get_example(id: usize, shard_progress: &ShardProgress) -> Example {
    assert!((shard_progress.used_examples..shard_progress.total_examples).contains(&id));

    let shard = shard_progress.shard.get();
    let examples = shard.examples().unwrap();

    // Should not happen but there is no control over this invariant in Rust.
    assert!(!examples.is_empty());

    let attributes = examples.get(id).attributes().unwrap();
    // TODO the byte vectors should be pre-allocated to ensure alignment of larger types.
    // Usually the alignment is at least 8 bytes and moreover NumPy can deal with
    // unaligned arrays (it is a slowdown).
    let mut result = vec![Vec::new(); attributes.len()];

    attributes.iter().map(|x| x.attribute_bytes().unwrap().to_vec()).collect()

    result
}

impl Iterator for ShardProgress {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        if self.used_examples >= self.total_examples {
            return None;
        }

        let res = get_example(self.used_examples, self);
        self.used_examples += 1;
        Some(res)
    }
}
