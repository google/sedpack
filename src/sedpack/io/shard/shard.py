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
"""Dataset shard manipulation.
"""

from pathlib import Path

import sedpack
from sedpack.io.metadata import DatasetStructure
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.types import ExampleT
from sedpack.io.shard.shard_writer_base import ShardWriterBase
from sedpack.io.shard.get_shard_writer import get_shard_writer


class Shard():
    """A shard contains N measurement pertaining to the same key"""

    def __init__(self, shard_info: ShardInfo,
                 dataset_structure: DatasetStructure,
                 dataset_root_path: Path) -> None:
        """Collect information about a new shard.

        Args:

            shard_info (ShardInfo): Information about this shard.

            dataset_structure (DatasetStructure): The structure of data being
            saved.

            dataset_root_path (Path): Path to the dataset.
        """
        # Information needed to save the shard.
        self.shard_info: ShardInfo = shard_info
        self.dataset_structure: DatasetStructure = dataset_structure
        self._dataset_path: Path = dataset_root_path

        self._shard_writer: ShardWriterBase | None = get_shard_writer(
            dataset_structure=dataset_structure,
            shard_file=self._get_full_path(),
        )

    def write(self, values: ExampleT) -> None:
        """Write an example on disk as TFRecord.

        Args:

            values (ExampleT): Attribute values.
        """
        if not self._shard_writer:
            raise ValueError("Attempting to write to a closed shard.")

        self._shard_writer.write(values)
        self.shard_info.number_of_examples += 1

    def close(self) -> ShardInfo:
        """Close shard and return statistics."""
        if self._shard_writer is None:
            raise ValueError("Closing a shard which has not been open.")

        self._shard_writer.close()
        self._shard_writer = None

        # Compute sha256 checksum.
        self.shard_info.file_infos[
            0].hash_checksums = self._compute_file_hash_checksums()

        # Return shard info.
        return self.shard_info

    def _get_full_path(self) -> Path:
        """Return full path to the shard file.
        """
        return self._dataset_path / self.shard_info.file_infos[0].file_path

    def _compute_file_hash_checksums(self) -> tuple[str, ...]:
        """Compute hash checksums of the shard file(-s).

        TODO This method should return a list of checksums defined by the user
        in `self.dataset_structure`.
        """
        # Compute sha256 checksum.
        shard_path = self._get_full_path()
        return sedpack.io.utils.hash_checksums(
            file_path=shard_path,
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )
