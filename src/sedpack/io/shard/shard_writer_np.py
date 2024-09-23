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
"""Dataset shard manipulation.

For information how to read and write TFRecord files see
https://www.tensorflow.org/tutorials/load_data/tfrecord
"""

from pathlib import Path

import numpy as np

from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import AttributeValueT, CompressionT, ExampleT
from sedpack.io.shard.shard_writer_base import ShardWriterBase


class ShardWriterNP(ShardWriterBase):
    """Shard writing capabilities.
    """

    def __init__(self, dataset_structure: DatasetStructure,
                 shard_file: Path) -> None:
        """Collect information about a new shard.

        Args:

            dataset_structure (DatasetStructure): The structure of data being
            saved.

            shard_file (Path): Full path to the shard file.
        """
        assert dataset_structure.shard_file_type == "npz"

        super().__init__(
            dataset_structure=dataset_structure,
            shard_file=shard_file,
        )

        self._buffer: dict[str, list[AttributeValueT]] = {}

    def _write(self, values: ExampleT) -> None:
        """Write an example on disk. Writing may be buffered.

        Args:

            values (dict[str, npt.NDArray[np.generic]]): Attribute values.
        """
        # Just buffer all values.
        if not self._buffer:
            self._buffer = {
                name: [np.copy(value)] for name, value in values.items()
            }
        else:
            for name, value in values.items():
                self._buffer[name].append(np.copy(value))

    def close(self) -> None:
        """Close the shard file(-s).
        """
        if not self._buffer:
            assert not self._shard_file.is_file()
            return

        # Write the buffer into a file.
        match self.dataset_structure.compression:
            case "ZIP":
                np.savez_compressed(str(self._shard_file),
                                    **self._buffer)  # type: ignore
            case "":
                np.savez(str(self._shard_file), **self._buffer)  # type: ignore
            case _:
                # Default should never happen since ShardWriterBase checks that
                # the requested compression type is supported.
                raise ValueError(f"Unsupported compression type "
                                 f"{self.dataset_structure.compression} in "
                                 f"ShardWriterNP, supported values are: "
                                 f"{self.supported_compressions()}")

        self._buffer = {}
        assert self._shard_file.is_file()

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return ["ZIP", ""]
