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
"""Information how shard files are saved."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from sedpack.io.file_info import FileInfo
from sedpack.io.types import HashChecksumT
from sedpack.io import utils


class ShardInfo(BaseModel):
    """Information about a single shard file and all examples stored in it.

    Attributes:

        file_infos (tuple[FileInfo, ...]): Each shard can be saved in multiple
        files to enable partial download and reading. Paths and hash checksums
        to the files for this shard relative to the dataset root dictionary.

        number_of_examples (int64): How many examples are stored in this shard
        file.

        custom_metadata (dict[str, any]): Custom metadata about examples in
        this shard.
    """
    file_infos: tuple[FileInfo, ...]
    number_of_examples: int = 0
    custom_metadata: dict[str, Any] = {}


class ShardListInfo(BaseModel):
    """Represents a ShardsList which has not been loaded yet from the
    corresponding info file.

    Attributes:

        shard_list_info_file (FileInfo): Path to the ShardsList file and its
        hash checksum. The path is relative to dataset $ROOT_DIRECTORY and the
        file name must be "shards_list.json". All children shards are contained
        in the same directory $DIR_NAME and all children shards lists are
        contained in subdirectories of $DIR_NAME.

        number_of_examples (int64): Number of examples that are children of
        this shard list. We save this in the parent ShardsList in order to
        enable indexing without the need to download and parse all children
        shard list files.

        number_of_shards (int64): Number of shards that are children of this
        shard list. We save this parameter in order to enable indexing into
        shards and to enable getting the number of shards that are going to be
        iterated.
    """
    shard_list_info_file: FileInfo
    number_of_examples: int = 0
    number_of_shards: int = 0

    @field_validator("shard_list_info_file")
    @classmethod
    def check_is_shards_list(cls, v: FileInfo) -> FileInfo:
        """Check the path is leading to `shards_list.json`.

        Args:

          cls: BaseModel validator has to be a classmethod.

          v (Path): The path to be checked.

        Returns: the original path `v`.

        Raises: ValueError in case the file name is not `shards_list.json`.
        """
        if v.file_path.name != "shards_list.json":
            raise ValueError(f"The name must be \"shards_list.json\", got "
                             f"{v.file_path.name} in the path {v.file_path}")
        return v


class ShardsList(BaseModel):
    """Represents a list of shards and is saved together with those (and
    possibly children ones).

    The top level ShardsLists are (possibly empty or not present): train, test,
    holdout, and possibly other lists defined by the user and retrievable by
    the Dataset.get_keras_split API.

    Saved in a directory (defined by the ShardListInfo) as a JSON file called
    `shards_list.json`.

    Attributes:

        relative_path_self (Path): Convenience path to the file where this
        structure is saved relative to the dataset root directory.

        number_of_examples (int64): Number of examples in all children shards
        together.

        shard_files (list[ShardInfo]): A list of all shards directly in this
        directory.

        children_shard_lists (list[ShardListInfo]): A dataclass containing
        information about direct children ShardsLists.
    """
    relative_path_self: Path
    number_of_examples: int = 0
    shard_files: list[ShardInfo] = []
    children_shard_lists: list[ShardListInfo] = []

    @field_validator("relative_path_self")
    @classmethod
    def check_is_shards_list(cls, v: Path) -> Path:
        """Check the path is leading to `shards_list.json` and if there is no
        directory traversal.

        Args:

          cls: BaseModel validator has to be a classmethod.

          v (Path): The path to be checked.

        Returns: the original path `v`.

        Raises: ValueError in case the file name is not `shards_list.json` or
        if there is ".." in the path.
        """
        if v.name != "shards_list.json":
            raise ValueError(f"The name must be \"shards_list.json\", got "
                             f"{v.name} in the path {v}")
        if ".." in v.parts:
            raise ValueError("A .. is present in the path which could allow "
                             "directory traversal above `dataset_root_path`.")
        return v

    def write_config(self, dataset_root_path: Path,
                     hashes: tuple[HashChecksumT, ...]) -> ShardListInfo:
        """Write this config into dataset root path / `relative_path_self` and
        return the corresponding `ShardListInfo`.

        Args:

            dataset_root_path (Path): Path where the dataset is saved.

            hashes (tuple[HashChecksumT, ...]): Which hashes to compute (can be
            empty if we do not need the output).

        Returns: an object of `ShardListInfo` representing this file.
        """
        file_info: FileInfo = utils.safe_update_file(
            dataset_root_path=dataset_root_path,
            relative_path=self.relative_path_self,
            info=self.model_dump_json(exclude_defaults=True, indent=2),
            hashes=hashes,
        )

        number_of_shards: int = len(self.shard_files) + sum(
            info.number_of_shards for info in self.children_shard_lists)

        return ShardListInfo(
            shard_list_info_file=file_info,
            number_of_examples=self.number_of_examples,
            number_of_shards=number_of_shards,
        )

    @staticmethod
    def load_or_create(dataset_root_path: Path,
                       relative_path_self: Path) -> "ShardsList":
        """Load or create a new object.

        Args:

            dataset_root_path (Path): Path where the dataset is saved.

            relative_path_self (Path): Path to this file.
        """
        canonical_path = (dataset_root_path / relative_path_self).resolve()

        if canonical_path.is_file():
            # Check that we do not walk up the `dataset_root_path` by accident.
            if not canonical_path.is_relative_to(dataset_root_path.resolve()):
                raise ValueError(f"Expected {canonical_path} to be relative "
                                 f"to {dataset_root_path.resolve()}")
            # Load
            return ShardsList.model_validate_json(
                (dataset_root_path / relative_path_self).read_text())

        return ShardsList(relative_path_self=relative_path_self)
