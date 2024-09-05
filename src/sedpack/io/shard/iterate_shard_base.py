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
"""Base class for shard iteration."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Generic, Iterable, TypeVar

from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import ExampleT

T = TypeVar("T")


class IterateShardBase(ABC, Generic[T]):
    """Remember everything to be able to iterate shards. This can be pickled
    and passed as a callable object into another process.
    """

    def __init__(self, dataset_structure: DatasetStructure,
                 process_record: Callable[[ExampleT], T] | None) -> None:
        """Initialize the shard iterator.

        Args:

          dataset_structure (DatasetStructure): The structure of the dataset
          allowing shard parsing.

          process_record (Callable[[ExampleT], T] | None): How to process each
          record. Needs to be pickleable (for multiprocessing).
        """
        self.dataset_structure: DatasetStructure = dataset_structure
        self.process_record: Callable[[ExampleT], T] | None = process_record

    @abstractmethod
    def iterate_shard(self, file_path: Path) -> Iterable[ExampleT]:
        """Iterate a shard.
        """

    @abstractmethod
    async def iterate_shard_async(self, file_path: Path):
        """Asynchronously iterate a shard.
        """

    @abstractmethod
    def process_and_list(self, shard_file: Path) -> list[T]:
        """Return a list of processed examples. Used as a function call in a
        different process. Returning a list as opposed to an iterator allows to
        do all work in another process and all that needs to be done is a
        memory copy between processes.

        Args:

            shard_file (Path): Path to the shard file.

        Returns: A list of examples present in the shard. If `process_record`
        is defined it is applied to all those examples.
        """
