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
from typing import get_args

from sedpack.io.shard_file_metadata import ShardsList, ShardListInfo
from sedpack.io.types import HashChecksumT, SplitT


def merge_shard_infos(
    updates: list[ShardListInfo],
    dataset_root: Path,
    common: int,
    hashes: tuple[HashChecksumT, ...],
) -> ShardListInfo:
    """Merge a list of new `ShardListInfo`s into the dataset.

    Args:

      updates (list[ShardListInfo]): New shards lists information to merge. All
      of these belonging to the same split.

      dataset_root (Path): Where the dataset is saved.

      common (int): A positive integer indicating how many directories deep are
      shared between all `ShardListInfo`s (relative from the `dataset_root`).
      All `updates` must share the first `common` path components (in
      particular, the same split at index 0). Setting `common=1` merges all the
      way up to `<split>/shards_list.json`; setting
      `common = 1 + len(relative_path_from_split.parts)` merges a worker's
      subtree up to `<split>/<relative_path_from_split>/shards_list.json`.

      hashes (tuple[HashChecksumT, ...]): A tuple of hash checksum algorithms.
    """
    if not updates:
        raise ValueError("Nothing to update.")
    if common < 1:
        raise ValueError(f"Expected common >= 1, got {common}")

    common_prefix = updates[0].shard_list_info_file.file_path.parts[:common]
    if len(common_prefix) < common:
        raise ValueError(
            f"Update path {updates[0].shard_list_info_file.file_path} is "
            f"shorter than common={common} directory components.")

    # Check that all common prefixes are the same.
    for update in updates:
        parts = update.shard_list_info_file.file_path.parts
        if len(parts) < common + 1:
            raise ValueError(
                f"Update path {update.shard_list_info_file.file_path} must "
                f"have at least {common + 1} path components.")
        if parts[:common] != common_prefix:
            raise ValueError(f"Not all relative paths are the same "
                             f"{parts[:common]} vs {common_prefix}")

    # Divide the updates to current level shard lists and the deeper ones.
    current_level: list[ShardListInfo] = [
        update for update in updates
        if len(update.shard_list_info_file.file_path.parts) == common + 1
    ]
    deeper_updates: list[ShardListInfo] = [
        update for update in updates
        if len(update.shard_list_info_file.file_path.parts) > common + 1
    ]
    assert len(current_level) + len(deeper_updates) == len(updates)

    # If there are no deeper updates below this directory and the single
    # ShardListInfo at this level already has the requested checksums computed,
    # return it directly without re-traversing or re-hashing its subtree.
    if (not deeper_updates and len(current_level) == 1 and
            len(current_level[0].shard_list_info_file.hash_checksums)
            == len(hashes)):
        return current_level[0]

    # The current level ShardsList on disk (already updated if current_level is
    # non-empty).
    root_shard_list: ShardsList = ShardsList.load_or_create(
        dataset_root_path=dataset_root,
        relative_path_self=Path().joinpath(*common_prefix) / "shards_list.json",
    )

    # Group deeper updates by their immediate child directory at depth `common`.
    recursively_update: defaultdict[str,
                                    list[ShardListInfo]] = defaultdict(list)
    for update in deeper_updates:
        current_path: Path = update.shard_list_info_file.file_path
        directory = str(current_path.parts[common])
        recursively_update[directory].append(update)

    # If any existing child directory has no deeper updates but lacks the
    # requested hash checksums, include it so its hashes are recomputed.
    for child in root_shard_list.children_shard_lists:
        child_dir = str(child.shard_list_info_file.file_path.parts[common])
        if (child_dir not in recursively_update and
                len(child.shard_list_info_file.hash_checksums) != len(hashes)):
            recursively_update[child_dir].append(child)

    # Recursively update only the child directories that changed.
    merged: dict[str, ShardListInfo] = {
        directory:
            merge_shard_infos(
                updates=recursive_updates,
                dataset_root=dataset_root,
                common=common + 1,
                hashes=hashes,
            ) for directory, recursive_updates in recursively_update.items()
    }

    # Preserve existing child directory ordering, replacing updated children in
    # place and appending newly created child directories afterwards.
    new_children: list[ShardListInfo] = []
    for child in root_shard_list.children_shard_lists:
        child_dir = str(child.shard_list_info_file.file_path.parts[common])
        if child_dir in merged:
            new_children.append(merged.pop(child_dir))
        else:
            new_children.append(child)
    new_children.extend(merged.values())
    root_shard_list.children_shard_lists = new_children

    root_shard_list.number_of_examples = (
        sum(s.number_of_examples for s in root_shard_list.shard_files) +
        sum(c.number_of_examples for c in root_shard_list.children_shard_lists))

    # Write the root_shard_list and return its ShardListInfo.
    return root_shard_list.write_config(
        dataset_root_path=dataset_root,
        hashes=hashes,
    )


def merge_splits_shard_infos(
        updates: list[ShardListInfo],
        dataset_root: Path,
        hashes: tuple[HashChecksumT, ...],
        relative_path_from_split: Path = Path("."),
) -> dict[SplitT, ShardListInfo]:
    """Group `updates` by split and merge each split up to
    `<split>/<relative_path_from_split>/shards_list.json`.

    Args:

      updates (list[ShardListInfo]): ShardListInfo updates across one or more
      splits.

      dataset_root (Path): Root directory of the dataset.

      hashes (tuple[HashChecksumT, ...]): Hash checksum algorithms to compute.

      relative_path_from_split (Path): Subdirectory relative to `<split>` up to
      which shard lists should be merged. When empty (`Path(".")`), merges all
      the way up to `<split>/shards_list.json`.

    Returns: Mapping from each updated `SplitT` to its merged `ShardListInfo`.
    """
    splits_to_update: defaultdict[SplitT,
                                  list[ShardListInfo]] = defaultdict(list)
    for info in updates:
        parts = info.shard_list_info_file.file_path.parts
        if len(parts) < 2:
            raise ValueError(
                f"Expected ShardListInfo path inside a split directory, got "
                f"{info.shard_list_info_file.file_path}")
        split = str(parts[0])
        if split not in get_args(SplitT):
            raise ValueError(f"Not a known split value: {split}")
        splits_to_update[split].append(info)  # type: ignore[index]

    common: int = 1 + len(relative_path_from_split.parts)
    return {
        split:
            merge_shard_infos(
                updates=split_updates,
                dataset_root=dataset_root,
                common=common,
                hashes=hashes,
            ) for split, split_updates in splits_to_update.items()
    }
