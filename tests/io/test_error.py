# Copyright 2023-2024 Google LLC
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

import sedpack
from sedpack.io import Dataset
from sedpack.io import Attribute, Metadata
from sedpack.io.errors import DatasetExistsError


def test_dataset_exists(tmpdir: str | Path) -> None:
    tiny_experiment_path: Path = Path(tmpdir) / "exists"
    # Metadata
    dataset_metadata = Metadata(description="Test of the lib")
    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="float32",
            shape=(10, ),
        ),
    ]
    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes)

    # First should be ok.
    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    with pytest.raises(DatasetExistsError):
        # This would overwrite.
        dataset = Dataset.create(
            path=tiny_experiment_path,
            metadata=dataset_metadata,
            dataset_structure=dataset_structure,
        )
