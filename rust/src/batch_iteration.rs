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

use rayon::prelude::*;
use tracing::{span, Level};

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
    shard_progress_iterator: Box<dyn Iterator<Item = ShardProgress> + Send>,
    shard_progress_cache: Vec<ShardProgress>,
    batch_size: usize,
    has_fixed_shape: Vec<bool>,
}

impl Batcher {
    pub fn new(
        files: Vec<ShardInfo>, threads: usize, batch_size: usize, has_fixed_shape: Vec<bool>,
    ) -> Self {
        let shard_progress_iterator =
            Box::new(parallel_map(|x| get_shard_progress(&x), files.into_iter(), threads));
        Batcher {
            shard_progress_iterator,
            shard_progress_cache: vec![],
            batch_size,
            has_fixed_shape,
        }
    }
}

#[derive(Debug)]
struct BatchingDataIndex {
    /// Which shard in BatchingData::shards.
    shards_index: usize,
    /// Which example in the concrete shard?
    example_index: usize,
}

/// Collected ShardProgress and pointers to which example from which shard should be used. This way
/// we move around pointers without copying from ShardProgress. The tradeoff in case of shuffling
/// is higher memory usage especially when a shard contains many examples and is much larger when
/// uncompressed and at the same time batch size is very large. The plan is to have two shuffling
/// modes -- one based on this code and another using a shuffle buffer.
///
/// When shuffling it is expected that each ShardProgress contributes only a couple of examples
/// (in the ideal case just one). This implementation is optimized for that case (thus not having
/// `BatchingData { shards_and_indexes: Vec<(ShardProgress, Vec<usize>)> }`.
struct BatchingData {
    shards: Vec<ShardProgress>,
    indexes: Vec<BatchingDataIndex>,
}

impl BatchingData {
    /// Create a batch.
    pub fn create_batch(&self, has_fixed_shape: &[bool]) -> Batch {
        let span = span!(Level::TRACE, "create_batch");
        let _enter = span.enter();

        has_fixed_shape
            .par_iter()
            .enumerate()
            .map(|(attribute_id, is_fixed)| {
                if *is_fixed {
                    self.get_static_attribute(attribute_id)
                } else {
                    self.get_dynamic_attribute(attribute_id)
                }
            })
            .collect()
    }

    fn get_static_attribute(&self, attribute_id: usize) -> BatchedAttribute {
        BatchedAttribute::Static {
            data: numpy::ndarray::Array::<u8, numpy::Ix1>::from_vec({
                let span = span!(Level::TRACE, "fill single static attribute");
                let _enter = span.enter();

                // Fixed len attribute.
                let attribute_len = self.shards[0].borrow_attribute(0, attribute_id).len();
                let batch_size = self.indexes.len();

                // Do memcpy in parallel.
                let mut v = vec![0; batch_size * attribute_len];
                v.par_chunks_exact_mut(attribute_len).enumerate().for_each(|(batch_i, slice)| {
                    slice.copy_from_slice({
                        let batching_index = &self.indexes[batch_i];
                        self.shards[batching_index.shards_index]
                            .borrow_attribute(batching_index.example_index, attribute_id)
                    });
                });
                v
            }),
        }
    }

    fn get_dynamic_attribute(&self, attribute_id: usize) -> BatchedAttribute {
        let span = span!(Level::TRACE, "fill single dynamic attribute");
        let _enter = span.enter();

        BatchedAttribute::Dynamic {
            data: self
                .indexes
                .par_iter()
                .map(|batching_index| {
                    numpy::ndarray::Array::<u8, numpy::Ix1>::from_vec(
                        self.shards[batching_index.shards_index]
                            .borrow_attribute(batching_index.example_index, attribute_id)
                            .to_vec(),
                    )
                })
                .collect(),
        }
    }
}

impl Iterator for Batcher {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let span = span!(Level::TRACE, "Batcher.next");
        let _enter = span.enter();

        // Select data for the batch to be filled.
        let mut batching_data = BatchingData { shards: Vec::new(), indexes: Vec::new() };
        while batching_data.indexes.len() < self.batch_size {
            match self.shard_progress_cache.pop() {
                None => {
                    // Refill if possible.
                    match self.shard_progress_iterator.next() {
                        Some(refill) => self.shard_progress_cache.push(refill),
                        None => {
                            break; // No refill.
                        }
                    }
                }
                Some(mut shard_progress) => {
                    let shards_index = batching_data.shards.len();

                    // Consume while we can or while the shard is not full.
                    let n_remaining_examples = self.batch_size - batching_data.indexes.len();
                    batching_data.indexes.par_extend(
                        shard_progress
                            .take_example_ids(n_remaining_examples)
                            .into_par_iter()
                            .map(|example_index| BatchingDataIndex { shards_index, example_index }),
                    );

                    // Also remember the shard to be used.
                    batching_data.shards.push(shard_progress);
                }
            };
        }

        if batching_data.shards.is_empty() {
            None
        } else {
            let current_batch = batching_data.create_batch(&self.has_fixed_shape);

            assert!(self.shard_progress_cache.is_empty());

            // Get back the shard progresses which are not fully used.
            self.shard_progress_cache.extend(
                batching_data.shards.into_iter().filter(|shard_progress| shard_progress.has_next()),
            );

            Some(current_batch)
        }
    }
}

pub struct BatchIterator {
    batch_iterator: Box<dyn Iterator<Item = Batch> + Send>,
}

impl BatchIterator {
    pub fn new(
        files: Vec<ShardInfo>, threads: usize, batch_size: usize, has_fixed_shape: Vec<bool>,
    ) -> Self {
        BatchIterator {
            batch_iterator: Box::new(Batcher::new(files, threads, batch_size, has_fixed_shape)),
        }
    }
}

impl Iterator for BatchIterator {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let span = span!(Level::TRACE, "BatchIterator.next");
        let _enter = span.enter();

        self.batch_iterator.next()
    }
}
