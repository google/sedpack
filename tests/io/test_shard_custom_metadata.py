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

from pathlib import Path
from typing import Any, Union

import numpy as np

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.types import TRAIN_SPLIT


def test_custom_shard_metadata(tmpdir: Union[str, Path]) -> None:
    attribute_dtype = "float32"
    shard_file_type = "fb"

    array_of_values = np.random.random((32, 138)).astype(dtype=attribute_dtype)

    experiment_path: Path = Path(tmpdir) / "custom_shard_metadata_experiment"

    # Create a dataset
    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=attribute_dtype,
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="GZIP",
        examples_per_shard=4,
        shard_file_type=shard_file_type,
    )

    dataset = Dataset.create(
        path=experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset
    custom_metadata_0 = {"key": {"key2": "valueA"}}
    custom_metadata_1 = {"key": {"key2": "valueB"}}

    with dataset.filler() as filler:
        # No custom metadata -> saved into its own shard (not overwritten by
        # subsequent custom_metadata writes).
        filler.write_example(
            values={"attribute_name": array_of_values[0]},
            split=TRAIN_SPLIT,
        )
        # Different custom_metadata -> opens a separate shard.
        filler.write_example(
            values={"attribute_name": array_of_values[1]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_0,
        )
        # Another shard opened for custom_metadata_1.
        filler.write_example(
            values={"attribute_name": array_of_values[2]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Still the same custom_metadata_1 shard.
        filler.write_example(
            values={"attribute_name": array_of_values[3]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Still the same custom_metadata_1 shard.
        filler.write_example(
            values={"attribute_name": array_of_values[4]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Reaches examples_per_shard=4 -> shard is closed immediately.
        filler.write_example(
            values={"attribute_name": array_of_values[5]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Opens a second shard for custom_metadata_1.
        filler.write_example(
            values={"attribute_name": array_of_values[6]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )

    # The object in memory and the saved metadata are the same.
    assert dataset._dataset_info == Dataset(experiment_path)._dataset_info

    # There are four shards: the full custom_metadata_1 shard (closed first),
    # followed by the three partial shards closed on exit.
    shards: list[ShardInfo] = list(dataset.shard_info_iterator(TRAIN_SPLIT))
    assert len(shards) == 4
    assert [(s.custom_metadata, s.number_of_examples) for s in shards] == [
        (custom_metadata_1, 4),
        ({}, 1),
        (custom_metadata_0, 1),
        (custom_metadata_1, 1),
    ]


def test_custom_metadata_none_and_present_saved_in_separate_shards(
    tmpdir: Union[str, Path],) -> None:
    """Examples without custom_metadata (`None` or `{}`) and examples with
    non-empty custom_metadata must be saved in separate shards regardless of
    write order.
    """
    values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    dataset_path = Path(tmpdir) / "none_vs_present_metadata"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test None vs non-empty custom_metadata"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="x",
                    dtype="float32",
                    shape=(8,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    meta_a = {"group": "A"}
    # Start with non-empty metadata, then None, then {}, then meta_a again.
    with dataset.filler(concurrency=2) as filler:
        filler.write_example(
            values={"x": values[0]},
            split=TRAIN_SPLIT,
            custom_metadata=meta_a,
        )
        filler.write_example(
            values={"x": values[1]},
            split=TRAIN_SPLIT,
            custom_metadata=None,
        )
        filler.write_example(
            values={"x": values[2]},
            split=TRAIN_SPLIT,
            custom_metadata={},
        )
        filler.write_example(
            values={"x": values[3]},
            split=TRAIN_SPLIT,
            custom_metadata=meta_a,
        )

    dataset.check(show_progressbar=False)
    shards = list(dataset.shard_info_iterator(TRAIN_SPLIT))
    assert len(shards) == 2

    shards_by_meta = {bool(s.custom_metadata): s for s in shards}
    assert shards_by_meta[False].custom_metadata == {}
    assert shards_by_meta[False].number_of_examples == 2
    assert shards_by_meta[True].custom_metadata == meta_a
    assert shards_by_meta[True].number_of_examples == 2

    no_meta_examples = [
        ex["x"] for ex in dataset.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
            shard_filter=lambda s: not s.custom_metadata,
        )
    ]
    meta_a_examples = [
        ex["x"] for ex in dataset.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
            shard_filter=lambda s: s.custom_metadata == meta_a,
        )
    ]
    np.testing.assert_allclose(np.stack(no_meta_examples), values[[1, 2]])
    np.testing.assert_allclose(np.stack(meta_a_examples), values[[0, 3]])


def test_oscillating_custom_metadata_packs_full_shards_within_concurrency(
    tmpdir: Union[str, Path],) -> None:
    """Oscillating custom_metadata across `<= concurrency` distinct values
    reuses open shards and packs full shards instead of creating 1-example
    shards, while never exceeding `concurrency` open shards.
    """
    # 3 metadata variants * 8 examples each = 24 examples; with
    # examples_per_shard=4 and concurrency=3 this should produce exactly
    # 6 full shards of 4 examples each (not 24 single-example shards).
    values = np.arange(24 * 4, dtype=np.float32).reshape(24, 4)
    dataset_path = Path(tmpdir) / "oscillating_metadata_full_shards"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test oscillating custom_metadata"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="x",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    metadatas: list[dict[str, Any] | None] = [
        None,
        {
            "id": 1,
            "label": "one"
        },
        {
            "label": "two",
            "id": 2
        },
    ]

    with dataset.filler(concurrency=3) as filler:
        for i, val in enumerate(values):
            filler.write_example(
                values={"x": val},
                split=TRAIN_SPLIT,
                custom_metadata=metadatas[i % 3],
            )
            assert len(filler._current_shards) <= 3

    dataset.check(show_progressbar=False)
    shards = list(dataset.shard_info_iterator(TRAIN_SPLIT))
    assert len(shards) == 6
    assert all(s.number_of_examples == 4 for s in shards)


def test_oscillating_custom_metadata_lru_eviction_and_mutation_safety(
    tmpdir: Union[str, Path],) -> None:
    """When oscillating across more distinct metadata values than
    `concurrency`, LRU eviction bounds `_current_shards` to at most
    `concurrency`, and in-place mutation of the caller's dict does not corrupt
    previously written shards.
    """
    values = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    dataset_path = Path(tmpdir) / "oscillating_lru_and_mutation"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test LRU eviction and mutation safety"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="x",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    mutable_meta: dict[str, Any] = {"nested": {"step": 0}}
    with dataset.filler(concurrency=2) as filler:
        # Write with step=0, then mutate mutable_meta in place to step=1 and
        # step=2. With concurrency=2, step=2 evicts the LRU shard (step=0).
        for step in [0, 1, 0, 1, 2]:
            mutable_meta["nested"]["step"] = step
            filler.write_example(
                values={"x": values[step]},
                split=TRAIN_SPLIT,
                custom_metadata=mutable_meta,
            )
            assert len(filler._current_shards) <= 2

    dataset.check(show_progressbar=False)
    shards = list(dataset.shard_info_iterator(TRAIN_SPLIT))
    # step=0 (2 examples, evicted when step=2 arrived), step=1 (2 examples),
    # step=2 (1 example).
    assert len(shards) == 3
    assert [(s.custom_metadata, s.number_of_examples) for s in shards] == [
        ({
            "nested": {
                "step": 0
            }
        }, 2),
        ({
            "nested": {
                "step": 1
            }
        }, 2),
        ({
            "nested": {
                "step": 2
            }
        }, 1),
    ]
