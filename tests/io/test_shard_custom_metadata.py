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
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata, DatasetStructure, Attribute
from sedpack.io.types import TRAIN_SPLIT


def test_custom_shard_metadata(tmpdir: Union[str, Path]) -> None:
    array_of_values = np.random.random((32, 138))

    experiment_path: Path = Path(tmpdir) / "custom_shard_metadata_experiment"

    # Create a dataset
    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="float32",
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="GZIP",
        examples_per_shard=4,
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
        # No custom metadata.
        filler.write_example(
            values={"attribute_name": array_of_values[0]},
            split=TRAIN_SPLIT,
        )
        # Still the same shard, retroactively setting metadata here.
        filler.write_example(
            values={"attribute_name": array_of_values[1]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_0,
        )
        # Another shard has been open.
        filler.write_example(
            values={"attribute_name": array_of_values[2]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Still the same shard.
        filler.write_example(
            values={"attribute_name": array_of_values[3]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Still the same shard.
        filler.write_example(
            values={"attribute_name": array_of_values[4]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Still the same shard.
        filler.write_example(
            values={"attribute_name": array_of_values[5]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )
        # Shard full, another opened.
        filler.write_example(
            values={"attribute_name": array_of_values[6]},
            split=TRAIN_SPLIT,
            custom_metadata=custom_metadata_1,
        )

    # The object in memory and the saved metadata are the same.
    assert dataset._dataset_info == Dataset(experiment_path)._dataset_info

    # There are three shards with the custom metadata.
    shards: list[ShardInfo] = list(dataset.shard_info_iterator(TRAIN_SPLIT))
    assert len(shards) == 3
    assert shards[0].custom_metadata == custom_metadata_0
    assert shards[1].custom_metadata == custom_metadata_1
    assert shards[2].custom_metadata == custom_metadata_1
