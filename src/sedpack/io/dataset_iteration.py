# Copyright 2024 Google LLC
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
"""Mixin for sedpack.io.Dataset to do iteration."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import itertools
import os
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Type,
)
from typing_extensions import Self

import asyncstdlib
import numpy as np
import tensorflow as tf

from sedpack.io.dataset_base import DatasetBase
from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.itertools import LazyPool
from sedpack.io.itertools import round_robin, round_robin_async, shuffle_buffer
from sedpack.io.npz import IterateShardNP
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.tfrec import IterateShardTFRec
from sedpack.io.tfrec.tfdata import get_from_tfrecord
from sedpack.io.types import ExampleT, ShardFileTypeT, SplitT, TFDatasetT

from sedpack import _sedpack_rs  # type: ignore[attr-defined]


class DatasetIteration(DatasetBase):
    """Mixin for sedpack.io.Dataset to do iteration.
    """

    def shard_paths_dataset(
        self,
        split: SplitT,
        shards: int | None = None,
        custom_metadata_type_limit: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
    ) -> list[str]:
        """Return a list of shard filenames.

        Args:

            split (SplitT): Split, see SplitT.

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

        Returns: A list of shards filenames.
        """
        # List of all shard informations
        shards_list: list[ShardInfo] = list(
            self.shard_info_iterator(split=split))

        # Filter which shards to use.
        if shard_filter is not None:
            shards_list = list(filter(shard_filter, shards_list))

            kept_metadata: set[str] = {
                str(s.custom_metadata) for s in shards_list
            }
            self._logger.info(
                "Filtered shards with custom metadata: %s from split: %s",
                kept_metadata,
                split,
            )

        # Check that there is still something to iterate
        if not shards_list:
            raise ValueError("The list of shards is empty. Try less "
                             "restrictive filtering.")

        # Truncate the shard list
        if shards:
            shards_list = shards_list[:shards]

        # Only use a limited amount of shards for each setting of
        # custom_metadata.
        if custom_metadata_type_limit:
            counts: dict[tuple[tuple[str, Any], ...], int] = {}
            old_shards_list = shards_list
            shards_list = []
            for shard_info in old_shards_list:
                k = tuple(sorted(shard_info.custom_metadata.items()))
                counts[k] = counts.get(k, 0) + 1
                if counts[k] <= custom_metadata_type_limit:
                    shards_list.append(shard_info)
            self._logger.info("Took %s shards total", len(shards_list))

        # Full shard file paths.
        shard_paths = [
            str(self.path / s.file_infos[0].file_path) for s in shards_list
        ]

        return shard_paths

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
                compression,  # type: ignore
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
        # Shard file names.
        shard_paths: list[str] = self.shard_paths_dataset(
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
        )

        # The user requested a tf.data.Dataset use as_numpy_iterator_concurrent
        # to provide.
        if self.dataset_structure.shard_file_type != "tfrec":
            output_signature = {
                attribute.name:
                    tf.TensorSpec(shape=attribute.shape, dtype=attribute.dtype)
                for attribute in self.dataset_structure.saved_data_description
            }
            tf_dataset = tf.data.Dataset.from_generator(
                lambda: self.as_numpy_iterator_concurrent(
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
                    process_record,  # type: ignore[arg-type]
                    num_parallel_calls=parallelism,
                )
            if shuffle:
                tf_dataset = tf_dataset.shuffle(shuffle)
            if batch_size > 0:
                # Batch
                tf_dataset = tf_dataset.batch(batch_size)
            return tf_dataset

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
                process_record,  # type: ignore[arg-type]
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

    async def as_numpy_iterator_async(
        self,
        *,
        split: SplitT,
        process_record: Callable[[ExampleT], T] | None = None,
        shards: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        file_parallelism: int = os.cpu_count() or 4,
        shuffle: int = 1_000,
    ) -> AsyncIterator[ExampleT] | AsyncIterator[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation. Iterates in random order.

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Callable[[ExampleT], T] | None): Optional
            function that processes a single record.

            shards (int | None): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Callable[[ShardInfo], bool | None): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            file_parallelism (int): IO parallelism.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self.as_numpy_common(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decode the files.
        supported_file_types: list[ShardFileTypeT] = ["npz", "fb"]
        if self.dataset_structure.shard_file_type not in supported_file_types:
            raise ValueError(f"The method as_numpy_iterator_async supports "
                             f"only {supported_file_types} but not "
                             f"{self.dataset_structure.shard_file_type}")

        shard_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case _:
                raise ValueError(f"{self.dataset_structure.shard_file_type} "
                                 f"is marked as supported but support is not "
                                 f"implemented.")

        # Automatically shuffle.
        if shuffle:
            example_iterator = round_robin_async(
                asyncstdlib.map(
                    shard_iterator.iterate_shard_async,  # type: ignore
                    shard_paths_iterator,
                ),
                buffer_size=file_parallelism,
            )
        else:
            example_iterator = asyncstdlib.chain.from_iterable(
                asyncstdlib.map(
                    shard_iterator.iterate_shard_async,  # type: ignore
                    shard_paths_iterator,
                ))

        # Process each record if requested.
        if process_record:
            example_iterator = asyncstdlib.map(process_record, example_iterator)

        async for example in example_iterator:
            yield example

    def as_numpy_common(
        self,
        *,
        split: SplitT,
        shards: int | None = None,
        custom_metadata_type_limit: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        shuffle: int = 1_000,
    ) -> Iterable[str]:
        """"Common part of as_numpy_iterator and as_numpy_iterator_*.

        Args:

            split (SplitT): Split, see SplitT.

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

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over shard paths.
        """
        # Shard file names.
        shard_paths: list[str] = self.shard_paths_dataset(
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
        )

        # Infinite loop over the shard paths
        if repeat:
            shard_paths_iterator = itertools.cycle(shard_paths)
        else:
            shard_paths_iterator = shard_paths  # type: ignore

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            shard_paths_iterator = shuffle_buffer(
                shard_paths_iterator,  # type: ignore
                buffer_size=len(shard_paths))
        return shard_paths_iterator

    def as_numpy_iterator_concurrent(
        self,
        *,
        split: SplitT,
        process_record: Callable[[ExampleT], T] | None = None,
        shards: int | None = None,
        custom_metadata_type_limit: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        file_parallelism: int = os.cpu_count() or 1,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation. Iterates in random order.

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

            file_parallelism (int): IO parallelism. Defaults to `os.cpu_count()
            or 1`.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self.as_numpy_common(
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decoding and processing function based on shard file type.
        # Since we do not know if `process_record` is applied or not we also do
        # not know the type of elements.
        shard_iterator: IterateShardBase[Any]
        match self.dataset_structure.shard_file_type:
            case "tfrec":
                shard_iterator = IterateShardTFRec(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                    num_parallel_calls=1,
                )
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                )
            case _:
                raise ValueError("Unsupported shard_file_type "
                                 f"{self.dataset_structure.shard_file_type}")

        # Do not use GPU with tfrecords to avoid allocating whole GPU memory by
        # each thread.
        if self.dataset_structure.shard_file_type == "tfrec":
            context = tf.device("CPU")
        else:
            context = contextlib.nullcontext()
        with context:
            if shuffle:
                with LazyPool(file_parallelism) as pool:
                    yield from round_robin(
                        pool.imap_unordered(
                            shard_iterator.process_and_list,  # type: ignore
                            shard_paths_iterator,
                        ),
                        # round_robin keeps the whole shard files in memory.
                        buffer_size=file_parallelism,
                    )
            else:
                # Iterate in order but avoid out of memory caused by eagerly
                # mapping a reading function across all shards.
                with ThreadPoolExecutor(
                        max_workers=file_parallelism) as executor:
                    # Do not iterate the start multiple times.
                    shard_paths_iterator = iter(shard_paths_iterator)
                    batch = list(
                        itertools.islice(shard_paths_iterator,
                                         file_parallelism))
                    while batch:
                        yield from itertools.chain.from_iterable(
                            executor.map(shard_iterator.process_and_list,
                                         batch))
                        batch = list(
                            itertools.islice(shard_paths_iterator,
                                             file_parallelism))

    def as_numpy_iterator(
        self,
        *,
        split: SplitT,
        process_record: Callable[[ExampleT], T] | None = None,
        shards: int | None = None,
        custom_metadata_type_limit: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation.

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

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self.as_numpy_common(
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decode the files.
        shard_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "tfrec":
                shard_iterator = IterateShardTFRec(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                    num_parallel_calls=os.cpu_count() or 1,
                )
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case _:
                raise ValueError(f"Unknown shard_file_type "
                                 f"{self.dataset_structure.shard_file_type}")

        example_iterator = itertools.chain.from_iterable(
            map(
                shard_iterator.iterate_shard,  # type: ignore
                shard_paths_iterator))  # type: ignore

        # Process each record if requested
        if process_record:
            example_iterator = map(process_record,
                                   example_iterator)  # type: ignore

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            example_iterator = shuffle_buffer(
                example_iterator,  # type: ignore
                buffer_size=shuffle)

        yield from example_iterator

    def as_numpy_iterator_rust(  # pylint: disable=too-many-arguments
        self,
        *,
        split: SplitT,
        process_record: Callable[[ExampleT], T] | None = None,
        shards: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        file_parallelism: int = os.cpu_count() or 1,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Experimental
        implementation using the Rust code.

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Callable[[ExampleT], T] | None): Optional
            function that processes a single record.

            shards (int | None): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Callable[[ShardInfo], bool | None): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            file_parallelism (int): IO parallelism. Defaults to `os.cpu_count()
            or 1`.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        # Only FlatBuffers are supported.
        if self.dataset_structure.shard_file_type != "fb":
            raise ValueError("This method is implemented only for FlatBuffers.")

        # Check if the compression type is supported by Rust.
        supported_compressions = _sedpack_rs.RustIter.supported_compressions()
        if self.dataset_structure.compression not in supported_compressions:
            raise ValueError(
                f"The compression {self.dataset_structure.compression} is not "
                "among the supported compressions: {supported_compressions}")

        with RustGenerator(
                dataset=self,
                split=split,
                process_record=process_record,
                shards=shards,
                shard_filter=shard_filter,
                repeat=repeat,
                file_parallelism=file_parallelism,
                shuffle=shuffle,
        ) as rust_generator:
            yield from rust_generator()


class RustGenerator:
    """A generator for tf.data.Dataset.from_generator which is reentrant (even
    when the iteration did not finished, which can happen when using
    tf.data.Dataset.from_generator).

    `_sedpack_rs.RustIter` is a Rust data structure which holds data unknown to
    Python reference counting (will be dropped at _sedpack_rs.RustIter.__exit__
    call (it could also implement droppable).

    `tf.data.Dataset.from_generator` expects a callable which returns an
    iterable. The problem is that it does call it even without exhausting the
    previous iterable. We need to prevent leaking data by creating multiple
    instances of `_sedpack_rs.RustIter`.

    The current implementation manages manually the context manager of
    `_sedpack_rs.RustIter` by calling __enter__ and __exit__. The current Rust
    implementation of `_sedpack_rs.RustIter` is reentrant.

    Possible solutions / alternatives:

    -   Wrap context manager similar to RustGenerator like follows:
        ```
        with RustGenerator(...) as gen:
          tf.data.Dataset.from_generator(gen, ...)
        ```

        The downside is that this is not very handy API (for training the user
        creates two datasets -> two indentation levels).

        The good part is that we are guaranteed to clean up.

    -   Depend on __del__ being called. With CPython this would be fine. But we
        would need to depend on TF dropping the reference (it should, not
        tested, but it should) which is not documented anywhere. The good part
        is nice API.

    -   Changing RustGenerator.__call__ into:

        ```
        def __call__(self):
          with _sedpack_rs.RustIter() as rust_iter:
            yield from rust_iter
        ```
        This should eventually clean all instances.

        The downside is that it is not obvious (implementation dependent) what
        happens when TF calls again (possibly from a different thread or
        process).
    """

    def __init__(self,
                 *,
                 dataset: DatasetIteration,
                 split: SplitT,
                 process_record: Callable[[ExampleT], T] | None = None,
                 shards: int | None = None,
                 shard_filter: Callable[[ShardInfo], bool] | None = None,
                 repeat: bool = True,
                 file_parallelism: int = os.cpu_count() or 1,
                 shuffle: int = 1_000) -> None:
        """A reentrant generator.

        Args:

          dataset (DatasetIteration): The dataset being iterated.

          split (SplitT): The split to be iterated.

          process_record (Callable[[ExampleT], T] | None): Optional
          transformation of each example.

          shards (int | None): Optional limit on the number of used shards.

          shard_filter (Callable[[ShardInfo], bool] | None): Optional predicate
          returning True for each shard which should be iterated.

          repeat (bool): Cycle infinitely.

          file_parallelism (int): How many files to read in parallel.

          shuffle (int): Size of the shuffle buffer.
        """
        self._rust_iter: _sedpack_rs.RustIter | None = None

        self._dataset: DatasetIteration = dataset
        self._split: SplitT = split
        self._process_record: Callable[[ExampleT], T] | None = process_record
        self._shards: int | None = shards
        self._shard_filter: Callable[[ShardInfo], bool] | None = shard_filter
        self._repeat: bool = repeat
        self._file_parallelism: int = file_parallelism
        self._shuffle: int = shuffle

        def to_dict(example: list[np.typing.NDArray[np.uint8]]) -> ExampleT:
            result: ExampleT = {}
            for np_bytes, attribute in zip(
                    example, dataset.dataset_structure.saved_data_description):
                result[attribute.name] = IterateShardFlatBuffer.decode_array(
                    np_bytes=np_bytes,
                    attribute=attribute,
                    batch_size=0,
                )
            return result

        self._to_dict = to_dict

    def __enter__(self) -> Self:
        """Enter the context manager (takes care of freeing memory held by
        Rust).
        """
        return self

    def __exit__(self, exc_type: Type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        """Drop the rust data structure holding content of open files and
        future examples.
        """
        if self._rust_iter is not None:
            self._rust_iter.__exit__(exc_type, exc_value, exc_tb)

    def __call__(self) -> Iterable[ExampleT] | Iterable[T]:
        """Return an iterable.
        """
        yield from self._single_iter()
        while self._repeat:
            yield from self._single_iter()

    def _single_iter(self) -> Iterable[ExampleT] | Iterable[T]:
        """Iterate the dataset once.
        """
        if self._rust_iter is None:
            shard_paths: list[str] = list(
                self._dataset.as_numpy_common(
                    split=self._split,
                    shards=self._shards,
                    shard_filter=self._shard_filter,
                    repeat=False,
                    shuffle=self._shuffle,
                ))

            self._rust_iter = _sedpack_rs.RustIter(
                files=shard_paths,
                repeat=False,
                threads=self._file_parallelism,
                compression=self._dataset.dataset_structure.compression,
            )
            # Manually calling __enter__ and __exit__ -- see class docstring.
            self._rust_iter.__enter__()  # pylint: disable=unnecessary-dunder-call
        elif not self._rust_iter.can_iterate:
            self._rust_iter.__enter__()  # pylint: disable=unnecessary-dunder-call

        example_iterator = map(self._to_dict, iter(self._rust_iter))
        if self._process_record:
            yield from map(self._process_record, example_iterator)
        else:
            yield from example_iterator

        self._rust_iter.__exit__(None, None, None)
        self._rust_iter = None
