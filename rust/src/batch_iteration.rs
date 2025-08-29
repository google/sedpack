// Copyright 2025 Google LLC
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

pub use super::example_iteration::{
    get_shard_progress, CompressionType, Example, ExampleIterator, ShardInfo, ShardProgress,
};
pub use super::parallel_map::parallel_map;
pub use super::shard_generated::sedpack::io::flatbuffer::shardfile::{root_as_shard, Shard};

/// Single attribute which has been batched.
pub enum BatchedAttribute {
    /// Row-major order batch of the attribute with static (fixed) size. That is in NumPy C-order
    /// we can index as data[batch_index][attribute_index] where batch_index in 0..batch_size and
    /// attribute_index in 0..len(attribute).
    Static { data: numpy::ndarray::Array<u8, numpy::Ix1> },
    /// Dynamic data where we do not know shape up front (e.g., string, bytearray) is represented
    /// as a vector with the same indexing semantic.
    Dynamic { data: Vec<numpy::ndarray::Array<u8, numpy::Ix1>> },
}

pub type Batch = Vec<BatchedAttribute>;

struct Batcher {
    example_iterator: Box<dyn Iterator<Item = Example> + Send>,
    batch_size: usize,
    has_fixed_shape: Vec<bool>,
}

impl Iterator for Batcher {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        // Collect examples.
        let cache: Vec<Example> = self.example_iterator.by_ref().take(self.batch_size).collect();

        // Decide if we have enough (the last batch might not have batch_size examples).
        if cache.is_empty() {
            return None;
        }

        // Batch the examples.
        let mut result = Batch::new();
        for (attribute_index, is_fixed) in self.has_fixed_shape.iter().enumerate() {
            // Collect batched version of current attribute across all cached examples.
            let current_batched_attribute = match is_fixed {
                true => BatchedAttribute::Static {
                    data: numpy::ndarray::Array::<u8, numpy::Ix1>::from_iter(
                        cache.iter().flat_map(|e| e[attribute_index].iter().cloned()),
                    ),
                },
                false => BatchedAttribute::Dynamic {
                    data: cache
                        .iter()
                        .map(|e| {
                            numpy::ndarray::Array::<u8, numpy::Ix1>::from_iter(
                                e[attribute_index].iter().cloned(),
                            )
                        })
                        .collect(),
                },
            };

            // Save the batched attribute.
            result.push(current_batched_attribute);
        }
        Some(result)
    }
}

pub struct BatchIterator {
    batch_iterator: Box<dyn Iterator<Item = Batch> + Send>,
}

impl BatchIterator {
    pub fn new(
        files: Vec<ShardInfo>, threads: usize, batch_size: usize, has_fixed_shape: Vec<bool>,
    ) -> Self {
        let example_iterator = Box::new(ExampleIterator::new(files, threads));
        let batch_iterator = Box::new(Batcher { example_iterator, batch_size, has_fixed_shape });
        BatchIterator { batch_iterator }
    }
}

impl Iterator for BatchIterator {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.batch_iterator.next()
    }
}
