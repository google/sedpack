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

pub use super::parallel_map::parallel_map;
pub use super::shard_generated::sedpack::io::flatbuffer::shardfile::{root_as_shard, Shard};

pub type Example = Vec<Vec<u8>>;
type LoadedShard = Yoke<Shard<'static>, Vec<u8>>;

/// Iterate all examples in given shard files.
pub struct ExampleIterator {
    example_iterator: Box<dyn Iterator<Item = Example> + Send>,
}

#[derive(Clone, Copy, Debug)]
pub enum CompressionType {
    Uncompressed,
    LZ4,
}

impl std::str::FromStr for CompressionType {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "" => Ok(CompressionType::Uncompressed),
            "LZ4" => Ok(CompressionType::LZ4),
            _ => Err("{input} unimplemented".to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShardInfo {
    pub file_path: String,
    pub compression_type: CompressionType,
}

impl ExampleIterator {
    /// Takes a vector of file names of shards and creates an ExampleIterator over those. We assume
    /// that all shard file names fit in memory. Alternatives to be re-evaluated:
    /// - Take an iterator passed from Python. That might require acquiring GIL and require
    ///   buffering.
    /// - Iterate over the shards in Rust. This would require having the shard filtering being
    ///   allowed to be called from Rust. But then we could pass an iterator of the following form:
    ///   `files: impl Iterator<Item = &str>`.
    pub fn new(files: Vec<ShardInfo>, repeat: bool, threads: usize) -> Self {
        assert!(!repeat, "Not implemented yet: repeat=true");
        let example_iterator = Box::new(
            parallel_map(|x| get_shard_progress(&x), files.into_iter(), threads).flatten(),
        );
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

/// Return a vector of bytes with the file content.
fn get_file_bytes(shard_info: &ShardInfo) -> Vec<u8> {
    match shard_info.compression_type {
        CompressionType::Uncompressed => std::fs::read(&shard_info.file_path).unwrap(),
        CompressionType::LZ4 => {
            let mut file_bytes = Vec::new();
            lz4_flex::frame::FrameDecoder::new(std::fs::File::open(&shard_info.file_path).unwrap())
                .read_to_end(&mut file_bytes)
                .unwrap();
            file_bytes
        }
    }
}

/// Get ShardProgress.
fn get_shard_progress(shard_info: &ShardInfo) -> ShardProgress {
    let file_bytes = get_file_bytes(shard_info);

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
fn get_example(id: usize, shard_progress: &ShardProgress) -> Example {
    assert!((shard_progress.used_examples..shard_progress.total_examples).contains(&id));

    let shard = shard_progress.shard.get();
    let examples = shard.examples().unwrap();

    // Should not happen but there is no control over this invariant in Rust.
    assert!(!examples.is_empty());

    let attributes = examples.get(id).attributes().unwrap();
    // TODO the byte vectors should be pre-allocated to ensure alignment of larger types. Usually
    // the alignment is at least 8 bytes and moreover NumPy can deal with unaligned arrays (it is
    // a slowdown).
    attributes.iter().map(|x| x.attribute_bytes().unwrap().iter().collect()).collect()
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
