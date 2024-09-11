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
"""Information how files are saved."""

from pathlib import Path

from pydantic import BaseModel, field_validator


class FileInfo(BaseModel):
    """Represents file information with a hash checksum(-s).

    Attributes:

        file_path (Path): The file path.  Relative to the dataset root
        directory.

        hash_checksums (tuple[str]): Control checksums of this shard file. In
        order given by DatasetStructure.hash_checksum_algorithms. As computed
        by `dataset_lib.io.utils.hash_checksums`.
    """
    file_path: Path
    hash_checksums: tuple[str, ...] = ()

    @field_validator("file_path")
    @classmethod
    def no_directory_traversal(cls, v: Path) -> Path:
        """Make sure there is no directory traversal.

        Args:

          cls: A validator for BaseModel needs to be a classmethod.

          v (Path): The path to be checked.

        Return: the original path `v`.

        Raises: ValueError in case `v` contains "..".
        """
        if ".." in v.parts:
            raise ValueError("A .. is present in the path which could allow "
                             "directory traversal above `dataset_root_path`.")
        return v
