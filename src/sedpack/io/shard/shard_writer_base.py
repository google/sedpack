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
"""Base class for shard writing depending on shard_file_type.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import ExampleT, CompressionT


class ShardWriterBase(ABC):
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
        # Information needed to save the shard.
        self.dataset_structure: DatasetStructure = dataset_structure
        self._shard_file: Path = shard_file

        # Make sure that the directory exists.
        self._shard_file.parent.mkdir(exist_ok=True, parents=True)

        # Make sure that the compression is supported for this shard file type.
        assert dataset_structure.compression in self.supported_compressions()

    def write(self, values: ExampleT) -> None:
        """Write an example on disk. Writing may be buffered.

        Args:

            values (ExampleT): Attribute values.
        """
        # Check the values are correct type and shape.
        for attribute in self.dataset_structure.saved_data_description:
            # If the attribute dtype is "bytes" and shape is empty tuple we
            # consider this a variable size attribute and do not check shape.
            if attribute.has_variable_size():
                continue

            # Else check the shape (the value should be a NumPy array but maybe
            # it is an int or bytearray).
            current_shape = np.array(values[attribute.name]).shape
            if current_shape != attribute.shape:
                raise ValueError(f"Attribute {attribute.name} has shape "
                                 f"{current_shape} expected {attribute.shape}")

        self._write(values=values)

    @abstractmethod
    def _write(self, values: ExampleT) -> None:
        """Write an example on disk. Writing may be buffered.

        Args:

            values (ExampleT): Attribute values.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the shard file(-s).
        """

    @staticmethod
    @abstractmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
