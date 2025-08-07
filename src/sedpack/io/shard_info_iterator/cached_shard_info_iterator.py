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
"""The default shard info iterator."""
import json
import logging
from pathlib import Path
import random

from typing import Callable, Iterator

from sedpack.io.metadata import DatasetInfo
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.shard_info_iterator.shard_info_iterator import ShardInfoIterator
from sedpack.io.types import SplitT


class CachedShardInfoIterator(ShardInfoIterator):
    """Iterate shards of a dataset.
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        dataset_info: DatasetInfo,
        split: SplitT | None,
        repeat: bool = False,
        shards: int | None = None,
        custom_metadata_type_limit: int | None = None,
        shard_filter: Callable[[ShardInfo], bool] | None = None,
        shuffle: int = 0,
    ) -> None:
        """Initialize shard information iteration.

        Args:

          dataset_path (Path): The path to the dataset directory.

          dataset_info (DatasetInfo): The information about the iterated
          dataset.

          split (SplitT | None): Which split to iterate or all if set to None.

          repeat (bool): Should we cycle indefinitely?

          shards (int | None): If specified limits the dataset to the first
          `shards` shards.

          custom_metadata_type_limit (int | None): Ignored when None. If
          non-zero then limit the number of shards with different
          `custom_metadata`. Take only the first `custom_metadata_type_limit`
          shards with the concrete `custom_metadata`. This is best effort for
          different `custom_metadata` (`json.dumps` with `sort_keys`).

          shard_filter (Callable[[ShardInfo], bool | None): If present this is
          a function taking the ShardInfo and returning True if the shard shall
          be used for traversal and False otherwise.

          shuffle (int): When set to 0 the iteration is deterministic otherwise
          shuffle the shards with a shuffle buffer of at least `shuffle`
          elements. Current implementation shuffles all shard information.
        """
        super().__init__(
            dataset_path=dataset_path,
            dataset_info=dataset_info,
            split=split,
            repeat=repeat,
        )

        self.shuffle: int = shuffle

        # Logging for non-trivial operations such as filtering custom metadata.
        self._logger = logging.getLogger("sedpack.io.Dataset")

        # Cache the list of shards.
        shard_list: list[ShardInfo] = list(
            ShardInfoIterator(
                dataset_path=dataset_path,
                dataset_info=dataset_info,
                split=split,
                repeat=False,
            ))

        # Filter if needed.
        if shard_filter:
            shard_list = [
                shard_info for shard_info in shard_list
                if shard_filter(shard_info)
            ]

            kept_metadata: set[str] = {
                json.dumps(
                    s.custom_metadata,
                    sort_keys=True,
                ) for s in shard_list
            }
            self._logger.info(
                "Filtered shards with custom metadata: %s from split: %s",
                kept_metadata,
                split,
            )

        # Only use a limited amount of shards for each setting of
        # custom_metadata.
        if custom_metadata_type_limit:
            counts: dict[str, int] = {}
            old_shards_list = shard_list
            shard_list = []
            for shard_info in old_shards_list:
                k: str = json.dumps(
                    shard_info.custom_metadata,
                    sort_keys=True,
                )
                counts[k] = counts.get(k, 0) + 1
                if counts[k] <= custom_metadata_type_limit:
                    shard_list.append(shard_info)
            self._logger.info("Took %s shards total", len(shard_list))

        # Limit the number of shards.
        if shards:
            shard_list = shard_list[:shards]

        # Initial shuffling.
        if shuffle:
            random.shuffle(shard_list)

        # Cached shards.
        self._index: int = -1  # The last returned element.
        self._shards: list[ShardInfo] = shard_list

    def number_of_shards(self) -> int:
        """Return the number of distinct shards that are iterated. When
        repeated this method still returns a finite answer.
        """
        return len(self._shards)

    def __iter__(self) -> Iterator[ShardInfo]:
        """Return the shard information iterator (reentrant).
        """
        return self

    def __next__(self) -> ShardInfo:
        """Get the next item.
        """
        self._index += 1

        if self._index >= len(self._shards):
            if self.repeat:
                self._index = 0
                if self.shuffle:
                    random.shuffle(self._shards)
            else:
                raise StopIteration

        return self._shards[self._index]
