# Copyright 2024-2025 Google LLC
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
"""Tests for custom dataset errors and disk-space checks."""

from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import sedpack
from sedpack.io import Dataset, DatasetFiller, Metadata
from sedpack.io.errors import DatasetExistsError, DiskFullError

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _create_test_dataset(
    dataset_path: Path,
    examples_per_shard: int = 4,
) -> Dataset:
    dataset_metadata = Metadata(description="Test of the lib")
    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="float32",
            shape=(10,),
        ),
    ]
    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        examples_per_shard=examples_per_shard,
    )
    return Dataset.create(
        path=dataset_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )


def test_dataset_exists(tmpdir: str | Path) -> None:
    tiny_experiment_path: Path = Path(tmpdir) / "exists"
    _create_test_dataset(tiny_experiment_path)

    with pytest.raises(DatasetExistsError):
        # This would overwrite.
        _create_test_dataset(tiny_experiment_path)


@pytest.mark.parametrize("invalid_utilization", [0.0, -0.1, 1.01, 2.0])
def test_filler_invalid_max_disk_utilization(
    tmpdir: str | Path,
    invalid_utilization: float,
) -> None:
    dataset = _create_test_dataset(Path(tmpdir) / "invalid_utilization")
    with pytest.raises(ValueError, match="max_disk_utilization"):
        dataset.filler(max_disk_utilization=invalid_utilization)


def test_filler_walks_up_to_existing_parent_and_zero_total(
    tmpdir: str | Path,) -> None:
    base_dir = Path(tmpdir)
    dataset = _create_test_dataset(base_dir / "existing_dataset")
    dataset.path = base_dir / "nested" / "not" / "created" / "yet"

    checked_paths: list[Path] = []

    def fake_disk_usage(path: Path) -> _DiskUsage:
        checked_paths.append(path)
        return _DiskUsage(total=0, used=0, free=0)

    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            side_effect=fake_disk_usage,
    ):
        filler = DatasetFiller(
            dataset=dataset,
            max_disk_utilization=0.8,
        )
        assert checked_paths == [base_dir]
        del filler


def test_filler_raises_at_creation_when_disk_full(tmpdir: str | Path) -> None:
    dataset = _create_test_dataset(Path(tmpdir) / "disk_full_at_creation")

    # 80% unavailable (including root-reserved blocks: used=750, free=200) ->
    # utilization is (1000 - 200) / 1000 = 0.8 >= 0.8.
    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            return_value=_DiskUsage(total=1000, used=750, free=200),
    ):
        with pytest.raises(DiskFullError) as exc_info:
            dataset.filler(max_disk_utilization=0.8)
        assert exc_info.value.current_utilization == pytest.approx(0.8)
        assert exc_info.value.max_disk_utilization == pytest.approx(0.8)

    # Default (max_disk_utilization=None) disables check even when free == 0.
    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            return_value=_DiskUsage(total=1000, used=1000, free=0),
    ):
        with dataset.filler() as filler:
            filler.write_example(
                values={"attribute_name": np.ones(10, dtype=np.float32)},
                split="train",
            )


def test_filler_saves_opened_shards_when_disk_fills_during_write(
    tmpdir: str | Path,) -> None:
    """When write_example raises DiskFullError before opening a new shard,
    DatasetFiller.__exit__ must still flush and save all so-far opened shards
    (both previously closed full shards and partial open shards across multiple
    splits) without losing data.
    """
    dataset = _create_test_dataset(
        Path(tmpdir) / "disk_full_mid_write",
        examples_per_shard=4,
    )

    train_data = np.arange(90, dtype=np.float32).reshape(9, 10)
    test_data = (np.arange(30, dtype=np.float32) + 1000.0).reshape(3, 10)

    current_usage = _DiskUsage(total=1000, used=500, free=500)

    def fake_disk_usage(unused_path: Path) -> _DiskUsage:
        _ = unused_path
        return current_usage

    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            side_effect=fake_disk_usage,
    ):
        with pytest.raises(DiskFullError):
            with dataset.filler(max_disk_utilization=0.8) as filler:
                # Write 8 train examples (2 full shards of 4).
                for i in range(8):
                    filler.write_example(
                        values={"attribute_name": train_data[i]},
                        split="train",
                    )
                # Write 3 test examples (1 partial open shard of 3).
                for i in range(3):
                    filler.write_example(
                        values={"attribute_name": test_data[i]},
                        split="test",
                    )

                # Simulate disk filling up to 85% before the 9th train example
                # (which attempts to open a 3rd train shard after the 2nd train
                # shard was closed on the 8th example).
                current_usage = _DiskUsage(total=1000, used=850, free=150)
                filler.write_example(
                    values={"attribute_name": train_data[8]},
                    split="train",
                )

        # Attempting to create a new filler while disk is still at 85% must
        # raise immediately at creation time.
        with pytest.raises(DiskFullError):
            dataset.filler(max_disk_utilization=0.8)

    # Verify dataset integrity and that all 8 train + 3 test examples were
    # saved without data loss.
    dataset.check(show_progressbar=False)
    reloaded = Dataset(dataset.path)
    reloaded.check(show_progressbar=False)

    saved_train = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split="train",
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(saved_train) == 8
    np.testing.assert_allclose(np.stack(saved_train), train_data[:8])

    saved_test = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split="test",
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(saved_test) == 3
    np.testing.assert_allclose(np.stack(saved_test), test_data)


def test_filler_disk_full_right_after_shard_rotation_does_not_double_close(
    tmpdir: str | Path,) -> None:
    """If disk utilization crosses max_disk_utilization after closing a full
    shard and before opening the next shard during write_example, __exit__
    must not close that shard a second time and the closed shard must be saved
    properly.
    """
    dataset = _create_test_dataset(
        Path(tmpdir) / "disk_full_after_rotation",
        examples_per_shard=2,
    )
    train_data = np.arange(30, dtype=np.float32).reshape(3, 10)

    # Calls to disk_usage:
    # 1. DatasetFiller.__init__ -> ok
    # 2. write_example #1 (before opening shard #1) -> ok
    # 3. write_example #3 (after shard #1 closed on #2, before opening shard #2)
    #    -> disk full (85%)!
    usages = iter([
        _DiskUsage(total=1000, used=500, free=500),
        _DiskUsage(total=1000, used=500, free=500),
        _DiskUsage(total=1000, used=850, free=150),
    ])

    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            side_effect=lambda unused_path: next(usages),
    ):
        with pytest.raises(DiskFullError):
            with dataset.filler(max_disk_utilization=0.8) as filler:
                for i in range(3):
                    filler.write_example(
                        values={"attribute_name": train_data[i]},
                        split="train",
                    )

    reloaded = Dataset(dataset.path)
    reloaded.check(show_progressbar=False)
    saved_train = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split="train",
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(saved_train) == 2
    np.testing.assert_allclose(np.stack(saved_train), train_data[:2])


def test_filler_disk_full_on_lru_eviction_saves_evicted_and_open_shards(
    tmpdir: str | Path,) -> None:
    """If LRU shard eviction closes a shard and DiskFullError is then raised
    before opening the new shard, both the evicted shard and any remaining open
    shards in the LRU cache are flushed and saved properly on __exit__.
    """
    dataset = _create_test_dataset(
        Path(tmpdir) / "disk_full_lru_eviction",
        examples_per_shard=4,
    )
    train_data = np.arange(30, dtype=np.float32).reshape(3, 10)

    # concurrency=2 allows 2 open shards in _current_shards.
    # Calls to disk_usage:
    # 1. DatasetFiller.__init__ -> ok
    # 2. write_example #1 (custom_metadata={"group": 0}) -> ok
    # 3. write_example #2 (custom_metadata={"group": 1}) -> ok
    # 4. write_example #3 (custom_metadata={"group": 2}, evicts group 0 and
    #    checks disk space before opening group 2) -> disk full (85%)!
    usages = iter([
        _DiskUsage(total=1000, used=500, free=500),
        _DiskUsage(total=1000, used=500, free=500),
        _DiskUsage(total=1000, used=500, free=500),
        _DiskUsage(total=1000, used=850, free=150),
    ])

    with patch(
            "sedpack.io.dataset_filler.shutil.disk_usage",
            side_effect=lambda unused_path: next(usages),
    ):
        with pytest.raises(DiskFullError):
            with dataset.filler(
                    concurrency=2,
                    max_disk_utilization=0.8,
            ) as filler:
                for i in range(3):
                    filler.write_example(
                        values={"attribute_name": train_data[i]},
                        split="train",
                        custom_metadata={"group": i},
                    )

    reloaded = Dataset(dataset.path)
    reloaded.check(show_progressbar=False)
    saved_train = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split="train",
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(saved_train) == 2
    np.testing.assert_allclose(np.stack(saved_train), train_data[:2])
