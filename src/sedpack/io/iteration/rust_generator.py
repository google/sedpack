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
import os
from types import TracebackType
from typing import (
    Callable,
    Iterable,
    Type,
)
from typing_extensions import Self

import numpy as np

from sedpack.io.dataset_base import DatasetBase
from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.types import ExampleT, SplitT

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

    def __init__(self,
                 *,
                 dataset: DatasetBase,
                 split: SplitT,
                 process_record: Callable[[ExampleT], T] | None = None,
                 shards: int | None = None,
                 shard_filter: Callable[[ShardInfo], bool] | None = None,
                 repeat: bool = True,
                 file_parallelism: int = os.cpu_count() or 1,
                 shuffle: int = 1_000) -> None:
        """A reentrant generator.

        Args:

          dataset (DatasetBase): The dataset being iterated.

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
        self._rust_iter: RustIter | None  # type: ignore[no-any-unimported]
        self._rust_iter = None

        self._dataset: DatasetBase = dataset
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

            self._rust_iter = RustIter(
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
