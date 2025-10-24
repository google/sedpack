# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional TensorFlow iteration."""

import os
from typing import Callable

from sedpack.io.dataset_base import DatasetBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.tfrec.tfdata import get_from_tfrecord
from sedpack.io.types import (
    ExampleT,
    SplitT,
    TFDatasetT,
)


class DatasetIterationTF(DatasetBase):
    """Implementation when TensorFlow is installed.
    """

    def read_and_decode(self, tf_dataset: TFDatasetT, cycle_length: int | None,
                        num_parallel_calls: int | None,
                        parallelism: int | None) -> TFDatasetT:
        """Read the shard files and decode them.

        Args:

          tf_dataset (tf.data.Dataset): Dataset containing shard paths as
          strings.

          cycle_length (int | None): How many files to read at once.

          num_parallel_calls (int | None): Number of parallel reading calls.

          parallelism (int | None): Decoding parallelism.

        Returns: tf.data.Dataset containing decoded examples.
        """
        # TensorFlow is an optional dependency.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        # If the cycle_length is None it is determined automatically but we do
        # use determinism. See documentation
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
        deterministic: bool | None = True
        if isinstance(cycle_length, int):
            # Use tf.data.Options.deterministic to decide `deterministic` if
            # cycle_length is <= 1.
            deterministic = False if cycle_length > 1 else None
        elif cycle_length is None:
            deterministic = True

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the fastest and most compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        #
        # If shuffle is equal to zero we produce deterministic order of
        # examples. By setting cycle_length to one (and num_parallel_calls to
        # default) we also avoid non-obvious interleaving patterns when there
        # are only a few shards. Deterministic defaults in order to avoid a
        # warning (deterministic = False does nothing when there is no thread
        # pool created by num_parallel_calls).
        tf_dataset = tf_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(
                x,
                compression_type=self.dataset_structure.
                compression,  # type: ignore[arg-type]
            ),
            cycle_length=cycle_length,
            block_length=1,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
        )

        # Decode to records
        tf_dataset = tf_dataset.map(
            get_from_tfrecord(self.dataset_structure.saved_data_description),
            num_parallel_calls=parallelism,
        )

        return tf_dataset

    def as_tfdataset(  # pylint: disable=too-many-arguments
            self,
            split: SplitT,
            *,
            process_record: Callable[[ExampleT], T] | None = None,
            shards: int | None = None,
            custom_metadata_type_limit: int | None = None,
            shard_filter: Callable[[ShardInfo], bool] | None = None,
            repeat: bool = True,
            batch_size: int = 32,
            prefetch: int = 2,
            file_parallelism: int | None = os.cpu_count(),
            parallelism: int | None = os.cpu_count(),
            shuffle: int = 1_000) -> TFDatasetT:
        """"Dataset as tfdataset

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Callable[[ExampleT], T] | None): Optional
            function that processes a single record.

            shards (int | None): If specified limits the dataset to the
            first `shards` shards.

            custom_metadata_type_limit (int | None): Ignored when None. If
            non-zero then limit the number of shards with different
            `custom_metadata`. Take only the first `custom_metadata_type_limit`
            shards with the concrete `custom_metadata`. This is best effort for
            different `custom_metadata` (hashed as a tuple of sorted items).

            shard_filter (Callable[[ShardInfo], bool | None): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            batch_size (int): Number of examples in a single batch. No batching
            if the `batch_size` is less or equal to zero.

            prefetch (int): Prefetch this many batches.

            file_parallelism (int | None): IO parallelism.

            parallelism (int | None): Parallelism of trace decoding and
            processing (ignored if shuffle is zero).

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: A tf.data.Dataset object of infinite stream of shuffled and
        batched examples.
        """
        # TensorFlow is an optional dependency.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        # The user requested a tf.data.Dataset use as_numpy_iterator_concurrent
        # to provide.
        if self.dataset_structure.shard_file_type != "tfrec":
            output_signature = {
                attribute.name:
                    tf.TensorSpec(shape=attribute.shape, dtype=attribute.dtype)
                for attribute in self.dataset_structure.saved_data_description
            }
            tf_dataset = tf.data.Dataset.from_generator(
                # The following method is provided by DatasetIteration.
                lambda: self.
                as_numpy_iterator_concurrent(  # type: ignore[attr-defined]
                    split=split,
                    process_record=None,  # otherwise unknown tensorspec
                    shards=shards,
                    shard_filter=shard_filter,
                    repeat=repeat,
                    file_parallelism=file_parallelism or 1,
                    shuffle=shuffle,
                ),
                output_signature=output_signature,
            )
            if process_record:
                tf_dataset = tf_dataset.map(
                    process_record,
                    num_parallel_calls=parallelism,
                )
            if shuffle:
                tf_dataset = tf_dataset.shuffle(shuffle)
            if batch_size > 0:
                # Batch
                tf_dataset = tf_dataset.batch(batch_size)
            return tf_dataset

        # The case when shard_file_type == "tfrec":
        # Shard file names.
        shard_paths: list[str] = self.shard_paths_dataset(  # type: ignore[attr-defined]
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
        )

        # Dataset creation
        tf_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)

        # Infinite loop over the shard paths
        if repeat:
            tf_dataset = tf_dataset.repeat()

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            tf_dataset = tf_dataset.shuffle(len(shard_paths))

        tf_dataset = self.read_and_decode(
            tf_dataset=tf_dataset,
            cycle_length=file_parallelism if shuffle else 1,
            num_parallel_calls=file_parallelism if shuffle else None,
            parallelism=parallelism,
        )

        # Process each record if requested
        if process_record:
            tf_dataset = tf_dataset.map(
                process_record,
                num_parallel_calls=parallelism,
            )

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            tf_dataset = tf_dataset.shuffle(shuffle)

        if batch_size > 0:
            # Batch
            tf_dataset = tf_dataset.batch(batch_size)

        # Prefetch (to keep loading when GPU is processing).
        tf_dataset = tf_dataset.prefetch(prefetch)

        return tf_dataset
