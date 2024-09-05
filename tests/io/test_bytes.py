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


def test_attribute_bytes(tmpdir: Union[str, Path]) -> None:
    array_of_values = [
        bytes(
            np.random.randint(
                0,  # low
                256,  # high
                np.random.randint(10, 1_000, (), np.int32),  # size
                np.uint8,
            )) for _ in range(138)
    ]

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="bytes",
            shape=(),  # ignored
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="GZIP",
        examples_per_shard=256,
    )

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset

    with dataset.filler() as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    # Check the data is correct

    for i, example in enumerate(
            dataset.as_tfdataset(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=0,
            )):
        if example["attribute_name"] != array_of_values[i]:
            print(example["attribute_name"])
            print(array_of_values[i])
            raise ValueError

    # We tested everything
    assert i + 1 == len(array_of_values)
