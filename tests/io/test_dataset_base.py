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
from typing import Union

import numpy as np

import sedpack
from sedpack.io import Dataset, Metadata


def test_dataset_info_is_copy(tmpdir: Union[str, Path]) -> None:
    dtype = "float32"
    compression = "LZ4"
    tiny_experiment_path: Path = Path(tmpdir) / "dataset_info"
    array_of_values = np.random.random((1024, 138))
    array_of_values = array_of_values.astype(dtype)

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=str(dtype),
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression=compression,
        examples_per_shard=256,
        shard_file_type="fb",
    )

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    old_dataset_info = dataset.dataset_info

    # Will make a copy
    dataset.dataset_info.dataset_structure.compression = ""

    assert old_dataset_info == dataset.dataset_info
