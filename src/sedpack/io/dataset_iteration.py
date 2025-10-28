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
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
)

import asyncstdlib
try:
    # TensorFlow is an optional dependency.
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore[assignment]

from sedpack.io.dataset_base import DatasetBase
from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.itertools import LazyPool
from sedpack.io.itertools import round_robin, round_robin_async, shuffle_buffer
from sedpack.io.npz import IterateShardNP
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.shard_info_iterator import CachedShardInfoIterator
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.tfrec import IterateShardTFRec
from sedpack.io.types import (
    BatchT,
    ExampleT,
    ShardFileTypeT,
    SplitT,
)
from sedpack.io.iteration import RustBatchedGenerator, RustGenerator


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
            different `custom_metadata` (hashed as `json.dumps`).

            shard_filter (Callable[[ShardInfo], bool | None): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

        Returns: A list of shards filenames.
        """
        shards_list: list[ShardInfo] = list(
            CachedShardInfoIterator(
                dataset_path=self.path,
                dataset_info=self.dataset_info,
                split=split,
                repeat=False,
                shards=shards,
                custom_metadata_type_limit=custom_metadata_type_limit,
                shard_filter=shard_filter,
                shuffle=0,
            ))

        # Check that there is still something to iterate
        if not shards_list:
            raise ValueError("The list of shards is empty. Try less "
                             "restrictive filtering.")

        # Full shard file paths.
        shard_paths = [
            str(self.path / s.file_infos[0].file_path) for s in shards_list
        ]

        return shard_paths

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
        # TODO(issue #85) Async iterator typing.
        example_iterator: AsyncIterator[ExampleT]
        if shuffle:
            example_iterator = round_robin_async(
                asyncstdlib.map(
                    shard_iterator.
                    iterate_shard_async,  # type: ignore[arg-type]
                    shard_paths_iterator,
                ),
                buffer_size=file_parallelism,
            )  # type: ignore[assignment]
        else:
            example_iterator = asyncstdlib.chain.from_iterable(
                asyncstdlib.map(
                    shard_iterator.
                    iterate_shard_async,  # type: ignore[arg-type]
                    shard_paths_iterator,
                ))

        # Process each record if requested.
        example_iterator_processed: AsyncIterator[ExampleT] | AsyncIterator[T]
        if process_record:
            example_iterator_processed = asyncstdlib.map(
                process_record, example_iterator)
        else:
            example_iterator_processed = example_iterator

        async for example in example_iterator_processed:
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
            shard_paths_iterator = shard_paths  # type: ignore[assignment]

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            shard_paths_iterator = shuffle_buffer(
                shard_paths_iterator,  # type: ignore[assignment]
                buffer_size=len(shard_paths),
            )
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
            if tf is None:
                raise ImportError("To use a TensorFlow record dataset please "
                                  "install TensorFlow")
            context = tf.device("CPU")
        else:
            context = contextlib.nullcontext()
        with context:
            if shuffle:
                with LazyPool(file_parallelism) as pool:
                    yield from round_robin(
                        pool.imap_unordered(
                            shard_iterator.
                            process_and_list,  # type: ignore[arg-type]
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
            `custom_metadata`. Take only the first
            `custom_metadata_type_limit` shards with the concrete
            `custom_metadata`. This is best effort for different
            `custom_metadata` (`json.dumps` with `sort_keys`).

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
        shard_iterator: Iterable[ShardInfo] = CachedShardInfoIterator(
            dataset_path=self.path,
            dataset_info=self.dataset_info,
            split=split,
            shards=shards,
            custom_metadata_type_limit=custom_metadata_type_limit,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        yield from self.example_iterator_from_shard_iterator(
            shard_iterator=shard_iterator,
            process_record=process_record,
            shuffle=shuffle,
        )

    def example_iterator_from_shard_iterator(
        self,
        *,
        shard_iterator: Iterable[ShardInfo],
        process_record: Callable[[ExampleT], T] | None = None,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """Low level iterator of examples given an iterator of shard
        information.

        Args:

          shard_iterator (Iterable[ShardInfo]): These shards are being
          iterated.

          process_record (Callable[[ExampleT], T] | None): Optional
          function that processes a single record.

          shuffle (int): How many examples should be shuffled across shards.
          When set to 0 the iteration is deterministic. It might be faster to
        """
        shard_paths_iterator: Iterable[str] = map(
            lambda shard_info: str(self.path / shard_info.file_infos[0].
                                   file_path),
            shard_iterator,
        )

        # Decode the files.
        shards_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "tfrec":
                shards_iterator = IterateShardTFRec(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                    num_parallel_calls=os.cpu_count() or 1,
                )
            case "npz":
                shards_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case "fb":
                shards_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case _:
                raise ValueError(f"Unknown shard_file_type "
                                 f"{self.dataset_structure.shard_file_type}")

        example_iterator = itertools.chain.from_iterable(
            map(
                shards_iterator.iterate_shard,  # type: ignore[arg-type]
                shard_paths_iterator,
            ))

        # Process each record if requested
        if process_record:
            example_iterator = map(
                process_record,
                example_iterator,  # type: ignore[assignment]
            )

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            example_iterator = shuffle_buffer(
                example_iterator,  # type: ignore[assignment]
                buffer_size=shuffle,
            )

        yield from example_iterator

    def as_numpy_iterator_rust_batched(  # pylint: disable=too-many-arguments
        self,
        *,
        split: SplitT,
        process_batch: Callable[[BatchT], T] | None = None,
        shards: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        repeat: bool = True,
        batch_size: int = 1,
        file_parallelism: int = os.cpu_count() or 1,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Experimental
        implementation using the Rust code.

        Args:

            split (SplitT): Split, see SplitT.

            process_batch (Callable[[BatchT], T] | None): Optional function
            that processes a batch of records.

            shards (int | None): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Callable[[ShardInfo], bool | None): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            batch_size (int): The batch size for RustBatchedGenerator.

            file_parallelism (int): IO parallelism. Defaults to `os.cpu_count()
            or 1`.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over batches (unless the parameter `process_batch`
        returns something else).
        """
        shard_iterator = CachedShardInfoIterator(
            dataset_path=self.path,
            dataset_info=self.dataset_info,
            split=split,
            repeat=repeat,
            shards=shards,
            custom_metadata_type_limit=None,
            shard_filter=shard_filter,
            shuffle=shuffle,
        )

        with RustBatchedGenerator(
                dataset_path=self.path,
                dataset_structure=self.dataset_structure,
                shard_iterator=shard_iterator,
                batch_size=batch_size,
                process_batch=process_batch,
                file_parallelism=file_parallelism,
        ) as rust_generator:
            yield from rust_generator()

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
        shard_iterator = CachedShardInfoIterator(
            dataset_path=self.path,
            dataset_info=self.dataset_info,
            split=split,
            repeat=repeat,
            shards=shards,
            custom_metadata_type_limit=None,
            shard_filter=shard_filter,
            shuffle=shuffle,
        )

        with RustGenerator(
                dataset_path=self.path,
                dataset_structure=self.dataset_structure,
                shard_iterator=shard_iterator,
                process_record=process_record,
                file_parallelism=file_parallelism,
        ) as rust_generator:
            yield from rust_generator()
