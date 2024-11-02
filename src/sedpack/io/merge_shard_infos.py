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
"""Merge new shard lists into the dataset.
"""

from collections import defaultdict
from pathlib import Path

from sedpack.io.shard_file_metadata import ShardsList, ShardListInfo
from sedpack.io.types import HashChecksumT


def merge_shard_infos(updates: list[ShardListInfo], dataset_root: Path,
                      common: int, hashes: tuple[HashChecksumT,
                                                 ...]) -> ShardListInfo:
    """Merge a list of new `ShardListInfo`s into the dataset.

    Args:

      updates (list[ShardListInfo]): New shards lists information to merge. All
      of these belonging to the same split.

      dataset_root (Path): Where the dataset is saved.

      common (int): A positive integer indicating how many directories deep are
      shared between all `ShardListInfo`s (relative from the `dataset_root`).
      When calling this function all `updates` have to be in the same split,
      thus one should set `common=1`. It is not guaranteed to update
      `shards_list.json` files all the way to the split when `common>1`.

      hashes (tuple[HashChecksumT, ...]): A tuple of hash checksum algorithms.
    """
    assert updates, "Nothing to update."

    # Check that all common prefixes are the same.
    for update in updates:
        if update.shard_list_info_file.file_path.parts[:common] != updates[
                0].shard_list_info_file.file_path.parts[:common]:
            raise ValueError(
                f"Not all relative paths are the same "
                f"{update.shard_list_info_file.file_path.parts[:common]} vs "
                f"{updates[0].shard_list_info_file.file_path.parts[:common]}")

    # The current level ShardsList (if it is in the updates).
    root_shard_list: ShardsList = ShardsList.load_or_create(
        dataset_root_path=dataset_root,
        relative_path_self=Path().joinpath(
            *updates[0].shard_list_info_file.file_path.parts[:common]) /
        "shards_list.json",
    )

    # Divide the updates to current level shard lists and the deeper ones.
    current_level: list[ShardListInfo] = [
        update for update in updates
        if len(update.shard_list_info_file.file_path.parts) == common + 1
    ]
    deeper_updates: list[ShardListInfo] = [
        update for update in updates
        if len(update.shard_list_info_file.file_path.parts) > common + 1
    ]
    # Check correctness of this implementation (O(1) check just to make sure we
    # do not forget anything).
    assert len(current_level) + len(deeper_updates) == len(updates)
    # Since the ShardsList is saved in a file named shards_list.json there can
    # be at most one update in this depth. We can ignore it since it has been
    # loaded into root_shard_list.
    assert len(current_level) <= 1

    # Move children of root_shard_list into deeper_updates to let recursion
    # merge everything.
    for child in root_shard_list.children_shard_lists:
        root_shard_list.number_of_examples -= child.number_of_examples
        deeper_updates.append(child)
    root_shard_list.children_shard_lists = []

    # Recursively update children with one longer common prefix.
    # Sort the infos by common directory.
    recursively_update: defaultdict[str,
                                    list[ShardListInfo]] = defaultdict(list)
    for update in deeper_updates:
        # The path is at least `split / $DIRECTORY / shards_list.json` or
        # longer.
        current_path: Path = update.shard_list_info_file.file_path
        directory = str(current_path.parts[common])
        recursively_update[directory].append(update)

    # Recursively update.
    merged: dict[str, ShardListInfo] = {  # Merge recursively.
        directory:
            merge_shard_infos(updates=recursive_updates,
                              dataset_root=dataset_root,
                              common=common + 1,
                              hashes=hashes)
        for directory, recursive_updates in recursively_update.items()
    }

    # Merge the recursive into root_shard_list.
    for child in merged.values():
        root_shard_list.number_of_examples += child.number_of_examples
        root_shard_list.children_shard_lists.append(child)

    # Write the root_shard_list and return its ShardListInfo.
    return root_shard_list.write_config(
        dataset_root_path=dataset_root,
        hashes=hashes,
    )
