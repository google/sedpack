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
"""Rust generator object wrapping the rust object to behave nicely with
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

from sedpack._sedpack_rs import RustIter


class RustGenerator:
    """A generator for tf.data.Dataset.from_generator which is reentrant (even
    when the iteration did not finished, which can happen when using
    tf.data.Dataset.from_generator).

    `RustIter` is a Rust data structure which holds data unknown to Python
    reference counting (will be dropped at RustIter.__exit__ call (it could also
    implement droppable).

    `tf.data.Dataset.from_generator` expects a callable which returns an
    iterable. The problem is that it does call it even without exhausting the
    previous iterable. We need to prevent leaking data by creating multiple
    instances of `RustIter`.

    The current implementation manages manually the context manager of
    `RustIter` by calling __enter__ and __exit__. The current Rust
    implementation of `RustIter` is reentrant.

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
          with RustIter() as rust_iter:
            yield from rust_iter
        ```
        This should eventually clean all instances.

        The downside is that it is not obvious (implementation dependent) what
        happens when TF calls again (possibly from a different thread or
        process).
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        dataset_structure: DatasetStructure,
        shard_iterator: Iterable[ShardInfo],
        process_record: Callable[[ExampleT], T] | None = None,
        file_parallelism: int = os.cpu_count() or 1,
    ) -> None:
        """A reentrant generator.

        Args:

          dataset_path (Path): The root path of the dataset.

          dataset_structure (DatasetStructure): The structure of the dataset.

          shard_iterator: (Iterable[ShardInfo]): How the shards should be
          iterated.

          process_record (Callable[[ExampleT], T] | None): Optional
          transformation of each example.

          file_parallelism (int): How many files to read in parallel.
        """
        self._rust_iter: RustIter | None  # type: ignore[no-any-unimported]
        self._rust_iter = None
        self._stopped: bool = False

        # Workaround until RustIter supports an Iterable[ShardInfo]. Take
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
        self._file_parallelism: int = file_parallelism

        # Only FlatBuffers are supported.
        if dataset_structure.shard_file_type != "fb":
            raise ValueError(
                "RustGenerator is implemented only for FlatBuffers.")

        # Check if the compression type is supported by Rust.
        supported_compressions = RustIter.supported_compressions()
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
                    batch_size=0,
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
        if self._rust_iter is not None:
            self._rust_iter.__exit__(exc_type, exc_value, exc_tb)

    def __call__(self) -> Iterable[ExampleT] | Iterable[T]:
        """Return an iterable.
        """
        while not self._stopped:
            yield from self._single_iter()

    def _single_iter(self) -> Iterable[ExampleT] | Iterable[T]:
        """Iterate over a single chunk of shards.
        """
        if self._rust_iter is None:
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

            self._rust_iter = RustIter(
                files=shard_paths,
                threads=self._file_parallelism,
                compression=self._dataset_structure.compression,
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
