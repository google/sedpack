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

from pathlib import Path
import pytest

from sedpack.io.file_info import FileInfo, PathGenerator
from sedpack.io.metadata import DatasetInfo
from sedpack.io.shard_file_metadata import ShardsList
from sedpack.io.shard_info_iterator import (
    CachedShardInfoIterator,
    ShardInfoIterator,
)
from sedpack.io.types import HOLDOUT_SPLIT
from sedpack.io.utils import safe_update_file


def test_multilevel() -> None:
    levels: int = 3
    max_branching: int = 2
    name_length: int = 3

    generator = PathGenerator(
        levels=levels,
        max_branching=max_branching,
        name_length=name_length,
    )

    seen_paths: set[Path] = set()

    # No exception in the top level
    for _ in range((max_branching**levels) + 10):
        p = generator.get_path()
        assert len(p.parts) == levels

        # Last part is long, previous are at most name_length
        assert len(p.parts[-1]) >= 32
        assert all(len(part) == name_length for part in p.parts[:-1])

        seen_paths.add(p)

    # Enforce bounded number of subdirectories except the top
    for l in range(levels - 1):
        for p in seen_paths:
            prefix = p.parents[l]
            count = sum(path.is_relative_to(prefix) for path in seen_paths)
            assert count <= max_branching**(l + 1)


def test_single_level() -> None:
    max_branching: int = 3
    name_length: int = 3
    generator = PathGenerator(
        levels=1,
        max_branching=max_branching,
        name_length=name_length,
    )

    # No exception in the top level
    for _ in range(max_branching + 10):
        n = generator.get_path()
        assert len(str(n)) >= 32  # last level is unbounded length
        assert len(n.parts) == 1


@pytest.mark.parametrize("bad_path", [
    Path("../escape/shard.fb"),
    Path("train/../../escape/shard.fb"),
    Path("/tmp/absolute_path.fb"),
])
def test_file_info_rejects_traversal_and_absolute_paths(bad_path: Path) -> None:
    """Verify `FileInfo` rejects '..' traversal and absolute paths."""
    with pytest.raises(ValueError, match="directory traversal"):
        FileInfo(file_path=bad_path)


@pytest.mark.parametrize("bad_path", [
    Path("../shards_list.json"),
    Path("/tmp/shards_list.json"),
])
def test_shards_list_rejects_traversal_and_absolute_paths(
    tmp_path: Path,
    bad_path: Path,
) -> None:
    """Verify `ShardsList` and `safe_update_file` delegate path sanitation to
    `FileInfo.no_directory_traversal` before touching disk.
    """
    with pytest.raises(ValueError, match="directory traversal"):
        ShardsList(relative_path_self=bad_path)
    with pytest.raises(ValueError, match="directory traversal"):
        ShardsList.load_or_create(
            dataset_root_path=tmp_path,
            relative_path_self=bad_path,
        )
    with pytest.raises(ValueError, match="directory traversal"):
        safe_update_file(
            dataset_root_path=tmp_path,
            relative_path=bad_path,
            info="{}",
            hashes=(),
        )


def test_empty_split_iterators(tmp_path: Path) -> None:
    """Verify iterating an unwritten split yields no shards even with
    `repeat=True`.
    """
    dataset_info = DatasetInfo()
    assert not list(
        ShardInfoIterator(
            dataset_path=tmp_path,
            dataset_info=dataset_info,
            split=HOLDOUT_SPLIT,
            repeat=False,
        ))
    assert not list(
        CachedShardInfoIterator(
            dataset_path=tmp_path,
            dataset_info=dataset_info,
            split=HOLDOUT_SPLIT,
            repeat=True,
        ))
