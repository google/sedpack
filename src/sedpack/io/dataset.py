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
"""Build and load tensorFlow dataset Record wrapper"""

from pathlib import Path
from typing import Union

from sedpack.io.dataset_base import DatasetBase
from sedpack.io.dataset_iteration import DatasetIteration
from sedpack.io.dataset_writing import DatasetWriting
from sedpack.io.errors import DatasetExistsError
from sedpack.io.metadata import DatasetInfo, DatasetStructure, Metadata


class Dataset(
        DatasetIteration,
        DatasetWriting,
):
    """Dataset class."""

    def __init__(self,
                 path: Union[str, Path],
                 create_dataset: bool = False) -> None:
        """Class for saving and loading a database.

        Args:

            path (Union[str, Path]): Path from where to load the dataset (a
            directory -- for instance
            "/home/user_name/datasets/my_awesome_dataset").

            create_dataset (bool): Are we creating a new dataset? Defaults to
            False which is used when (down-)loading a dataset.

        Raises:

            ValueError if the dataset was created using a newer version of the
            sedpack than the one trying to load it. See
            sedpack.__version__ docstring.

            FileNotFoundError if `create_dataset` is False and the
            `dataset_info.json` file does not exist.
        """
        dataset_info: DatasetInfo
        if create_dataset:
            # Default DatasetInfo.
            dataset_info = DatasetInfo()
        else:
            # Load the information.
            dataset_info = DatasetBase._load(Path(path))

        super().__init__(path=path, dataset_info=dataset_info)

    @staticmethod
    def create(
        path: Union[str, Path],
        metadata: Metadata,
        dataset_structure: DatasetStructure,
    ) -> "Dataset":
        """Create an empty dataset to be filled using the `filler` or
        `write_multiprocessing` API.

        Args:

            path (Union[str, Path]): Path where the dataset gets saved (a
            directory -- for instance
            "/home/user_name/datasets/my_awesome_dataset").

            metadata (Metadata): Information about this dataset.

            dataset_structure (DatasetStructure): Structure of saved records.

        Raises: DatasetExistsError if creating this object would overwrite the
        corresponding config file.
        """
        # Create a new object.
        dataset = Dataset(path=Path(path), create_dataset=True)

        # Do not overwrite an existing dataset.
        if Dataset._get_config_path(dataset.path).is_file():
            # Raise if the dataset already exists.
            raise DatasetExistsError(dataset.path)

        # Create a new dataset directory if needed.
        dataset.path.mkdir(parents=True, exist_ok=True)

        # Fill metadata and structure parameters.
        dataset.metadata = metadata
        dataset.dataset_structure = dataset_structure

        # Write empty config.
        dataset.write_config(updated_infos=[])
        return dataset
