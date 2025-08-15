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
"""Base class for a shard info iterator."""
import itertools
from pathlib import Path

from typing import Iterator

from sedpack.io.metadata import DatasetInfo
from sedpack.io.shard_file_metadata import ShardInfo, ShardsList, ShardListInfo
from sedpack.io.types import SplitT


class ShardInfoIterator:
    """Iterate shards of a dataset.
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        dataset_info: DatasetInfo,
        split: SplitT | None,
        repeat: bool = False,
    ) -> None:
        """Initialize shard information iteration.

        Args:

          dataset_path (Path): The path to the dataset directory.

          dataset_info (DatasetInfo): The information about the iterated
          dataset.

          split (SplitT | None): Which split to iterate or all if set to None.

          repeat (bool): Should we cycle indefinitely?
        """
        self.dataset_path: Path = Path(dataset_path)
        self.dataset_info: DatasetInfo = dataset_info
        self.split: SplitT | None = split
        self.repeat: bool = repeat

        self._iterator: Iterator[ShardInfo] = iter([])

    def __len__(self) -> int:
        """Either return the number of ShardInfo objects iterated or raise a
        ValueError if infinite cycle.
        """
        if self.number_of_shards() == 0 or not self.repeat:
            return self.number_of_shards()
        raise ValueError("Infinite iteration")

    def number_of_shards(self) -> int:
        """Return the number of distinct shards that are iterated. When
        repeated this method still returns a finite answer.
        """
        # Single split.
        if self.split is None:
            # Sum all splits.
            return sum(shard_list_info.number_of_shards
                       for shard_list_info in self.dataset_info.splits.values())

        if self.split not in self.dataset_info.splits:
            return 0

        return self.dataset_info.splits[self.split].number_of_shards

    def _shard_info_iterator(
            self, shard_list_info: ShardListInfo) -> Iterator[ShardInfo]:
        """Recursively yield `ShardInfo` from the whole directory tree.
        """
        shard_list: ShardsList = ShardsList.model_validate_json(
            (self.dataset_path /
             shard_list_info.shard_list_info_file.file_path).read_text())

        yield from shard_list.shard_files

        for child in shard_list.children_shard_lists:
            yield from self._shard_info_iterator(child)

    def __iter__(self) -> Iterator[ShardInfo]:
        """Return the shard information iterator (reentrant).
        """
        if self.split is None:
            self._iterator = itertools.chain.from_iterable(
                self._shard_info_iterator(shard_list_info)
                for shard_list_info in self.dataset_info.splits.values())
        else:
            self._iterator = self._shard_info_iterator(
                self.dataset_info.splits[self.split])

        return self._iterator

    def __next__(self) -> ShardInfo:
        """Get the next item.
        """
        return next(self._iterator)
