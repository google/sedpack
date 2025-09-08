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
"""Rust batched generator object wrapping the rust object to behave nicely with
TensorFlow."""
import itertools
import os
from pathlib import Path
from types import TracebackType
from typing import (
    Callable,
    Iterable,
    Iterator,
    Type,
)
from typing_extensions import Self

import numpy as np

from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.metadata import DatasetStructure
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.types import ExampleT

from sedpack._sedpack_rs import BatchedRustIter


class RustBatchedGenerator:
    """Similar to sedpack.io.iteration.RustGenerator with batching.
    Experimental API, expect breaking changes.
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        dataset_structure: DatasetStructure,
        shard_iterator: Iterable[ShardInfo],
        batch_size: int,
        process_record: Callable[[ExampleT], T] | None = None,
        file_parallelism: int = os.cpu_count() or 1,
    ) -> None:
        """A reentrant generator.

        Args:

          dataset_path (Path): The root path of the dataset.

          dataset_structure (DatasetStructure): The structure of the dataset.

          shard_iterator: (Iterable[ShardInfo]): How the shards should be
          iterated.

          batch_size (int): Size of the batches.

          process_record (Callable[[ExampleT], T] | None): Optional
          transformation of each example.

          file_parallelism (int): How many files to read in parallel.
        """
        self._iter: BatchedRustIter | None  # type: ignore[no-any-unimported]
        self._iter = None
        self._stopped: bool = False

        # Workaround until BatchedRustIter supports an Iterable[ShardInfo]. Take
        # _shard_chunk_size shard paths at once.
        self._shard_chunk_size: int = 1_000_000

        # Check file_parallelism is positive.
        if file_parallelism <= 0:
            raise ValueError("The argument file_parallelism should be "
                             f"positive but is {file_parallelism}")

        self._dataset_path: Path = dataset_path
        self._dataset_structure: DatasetStructure = dataset_structure
        # Make sure that any iteration on shard_iterator advances instead of
        # starting again.
        self._shard_iterator: Iterator[ShardInfo] = iter(shard_iterator)
        self._process_record: Callable[[ExampleT], T] | None = process_record
        self._batch_size: int = batch_size
        self._file_parallelism: int = file_parallelism

        # Which attributes have fixed shapes and which do not.
        self._has_fixed_shape: tuple[bool, ...] = tuple(
            not attribute.has_variable_size()
            for attribute in dataset_structure.saved_data_description)

        # Only FlatBuffers are supported.
        if dataset_structure.shard_file_type != "fb":
            raise ValueError(
                "RustBatchedGenerator is implemented only for FlatBuffers.")

        # Check if the compression type is supported by Rust.
        supported_compressions = BatchedRustIter.supported_compressions()
        if dataset_structure.compression not in supported_compressions:
            raise ValueError(
                f"The compression {dataset_structure.compression} is not "
                "among the supported compressions: {supported_compressions}")

        def to_dict(example: list[np.typing.NDArray[np.uint8]]) -> ExampleT:
            result: ExampleT = {}
            for np_bytes, attribute in zip(
                    example, dataset_structure.saved_data_description):
                result[attribute.name] = IterateShardFlatBuffer.decode_array(
                    np_bytes=np_bytes,
                    attribute=attribute,
                    batch_size=-1,
                )
            return result

        self._to_dict = to_dict

    def __enter__(self) -> Self:
        """Enter the context manager (takes care of freeing memory held by
        Rust).
        """
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Drop the rust data structure holding content of open files and
        future examples.
        """
        if self._iter is not None:
            self._iter.__exit__(exc_type, exc_value, exc_tb)

    def __call__(self) -> Iterable[ExampleT] | Iterable[T]:
        """Return an iterable.
        """
        while not self._stopped:
            yield from self._single_iter()

    def _single_iter(self) -> Iterable[ExampleT] | Iterable[T]:
        """Iterate over a single chunk of shards.
        """
        if self._iter is None:
            shard_paths: list[str] = [
                str(self._dataset_path / s.file_infos[0].file_path)
                for s in itertools.islice(
                    self._shard_iterator,
                    self._shard_chunk_size,
                )
            ]

            if not shard_paths:
                # No shards to iterate.
                self._stopped = True
                return

            self._iter = BatchedRustIter(
                files=shard_paths,
                threads=self._file_parallelism,
                compression=self._dataset_structure.compression,
                batch_size=self._batch_size,
                has_fixed_shape=self._has_fixed_shape,
            )
            # Manually calling __enter__ and __exit__ -- see class docstring.
            self._iter.__enter__()  # pylint: disable=unnecessary-dunder-call
        elif not self._iter.can_iterate:
            self._iter.__enter__()  # pylint: disable=unnecessary-dunder-call

        example_iterator = map(self._to_dict, iter(self._iter))
        if self._process_record:
            yield from map(self._process_record, example_iterator)
        else:
            yield from example_iterator

        self._iter.__exit__(None, None, None)
        self._iter = None
