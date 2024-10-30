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
"""Base class for a dataset."""
import logging
from pathlib import Path
import semver

from typing import Iterator, Union

import sedpack
from sedpack.io.shard_file_metadata import ShardInfo, ShardsList, ShardListInfo
from sedpack.io.types import SplitT
from sedpack.io.metadata import DatasetInfo, DatasetStructure, Metadata


class DatasetBase:
    """Base class for the dataset. Holds information and allows iteration over
    ShardInfo.
    """

    def __init__(self, path: Union[str, Path],
                 dataset_info: DatasetInfo) -> None:
        """Initialize the data.

        Args:

            path (Union[str, Path]): Path from where to load the dataset (a
            directory -- for instance
            "/home/user_name/datasets/my_awesome_dataset").
        """
        # Ensure pathlib Path.
        self.path: Path = Path(path)
        try:
            # Expand user directory.
            self.path = self.path.expanduser()
        except RuntimeError:
            # Expansion failed we assume that the path was not supposed to be
            # expanded.
            pass
        # Resolve the dataset root path to avoid problems when checking if
        # another path is relative to it.
        self.path = self.path.resolve()

        # Default DatasetInfo.
        self._dataset_info = dataset_info

        self._logger = logging.getLogger("sedpack.io.Dataset")

    @staticmethod
    def _load(path: Path) -> DatasetInfo:
        """Load a dataset from a config file.

        Args:

            path (Path): The path to the dataset.

        Raises:

            ValueError if the dataset was created using a newer version of the
            sedpack than the one trying to load it. See
            sedpack.__version__ docstring.

            FileNotFoundError if the `dataset_info.json` file does not exist.

        Returns: A DatasetInfo object representing the dataset metadata.
        """
        dataset_info = DatasetInfo.model_validate_json(
            DatasetBase._get_config_path(path).read_text(encoding="utf-8"))

        # Check that the library version (version of this software) is not
        # lower than what was used to capture the dataset.
        if semver.Version.parse(dataset_info.metadata.sedpack_version).compare(
                sedpack.__version__) > 0:
            raise ValueError(f"Dataset-lib module is outdated, "
                             f"sedpack_version: {sedpack.__version__}, "
                             f"but dataset was created using: "
                             f"{dataset_info.metadata.sedpack_version}")

        return dataset_info

    @staticmethod
    def _get_config_path(path: Path, relative: bool = False) -> Path:
        """Get the dataset config path.

        Args:

          path (Path): The dataset path.

          relative (bool): Return only relative to `path`. Defaults to False.

        Returns: A path to the config.
        """
        relative_path: Path = Path("dataset_info.json")
        if relative:
            return relative_path
        return path / relative_path

    @property
    def metadata(self) -> Metadata:
        """Return the metadata of this dataset.
        """
        return self._dataset_info.metadata

    @metadata.setter
    def metadata(self, value: Metadata) -> None:
        """Set the metadata of this dataset.
        """
        self._dataset_info.metadata = value

    @property
    def dataset_structure(self) -> DatasetStructure:
        """Return the structure of this dataset.
        """
        return self._dataset_info.dataset_structure

    @dataset_structure.setter
    def dataset_structure(self, value: DatasetStructure) -> None:
        self._dataset_info.dataset_structure = value

    def _shard_info_iterator(
            self, shard_list_info: ShardListInfo) -> Iterator[ShardInfo]:
        """Recursively yield `ShardInfo` from the whole directory tree.
        """
        shard_list: ShardsList = ShardsList.model_validate_json(
            (self.path /
             shard_list_info.shard_list_info_file.file_path).read_text())

        yield from shard_list.shard_files

        for child in shard_list.children_shard_lists:
            yield from self._shard_info_iterator(child)

    def shard_info_iterator(self, split: SplitT | None) -> Iterator[ShardInfo]:
        """Iterate all `ShardInfo` in the split.

        Args:

          split (SplitT | None): Which split to iterate shard information from.
          If None then all splits are iterated.

        Raises: ValueError when the split is not present. A split not being
        present is different from there not being any shard.
        """
        if split:
            if split not in self._dataset_info.splits:
                # Split not present.
                raise ValueError(f"There is no shard in {split}.")

            shard_list_info: ShardListInfo = self._dataset_info.splits[split]

            yield from self._shard_info_iterator(shard_list_info)
        else:
            for shard_list_info in self._dataset_info.splits.values():
                yield from self._shard_info_iterator(shard_list_info)
