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
        if repeat {
            panic!("Not implemented yet: repeat=true");
        }
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
fn get_shard_progress(file_path: String) -> ShardProgress {
    // TODO compressed file support.
    let mut file = match std::fs::File::open(file_path) {
        Ok(f) => f,
        Err(_e) => panic!("panic"), // No file no shard.
    };
    let mut file_bytes = Vec::new();
    let _ = file.read_to_end(&mut file_bytes);

    // A shard is a vector of examples (positive number -- invariant kept by Python code).
    // An example is vector of attributes (the same number of attributes in each example of each
    // shard).
    // An attribute is a vector of bytes.
    // For more details see the FlatBuffers schema.
    //
    // If parsing fails at any time it fails.
    let shard: LoadedShard = Yoke::attach_to_cart(file_bytes, |x| root_as_shard(x).unwrap());
    // Number of examples might be different in different shards.
    let examples = shard.get().examples().unwrap();

    ShardProgress { total_examples: examples.len(), used_examples: 0, shard }
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
    assert!(id >= shard_progress.used_examples);
    assert!(id < shard_progress.total_examples);

    let shard = shard_progress.shard.get();
    let examples = shard.examples().unwrap();

    // Should not happen but there is no control over this invariant in Rust.
    assert!(!examples.is_empty());

    let attributes = examples.get(id).attributes().unwrap();
    // TODO the byte vectors should be pre-allocated to ensure alignment of larger types.
    // Usually the alignment is at least 8 bytes and moreover NumPy can deal with
    // unaligned arrays (it is a slowdown).
    let mut result = Vec::new();
    result.resize(attributes.len(), Vec::new());

    // Parse and save examples.
    for (attribute_id, result_attribute) in result.iter_mut().enumerate() {
        let attribute_bytes = attributes.get(attribute_id).attribute_bytes().unwrap();

        // This is a memory copy. Thus `result` outlives `shard_progress`.
        result_attribute.extend(attribute_bytes);
    }

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
