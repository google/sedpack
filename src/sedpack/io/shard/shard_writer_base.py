# Copyright 2024-2025 Google LLC
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

import sedpack
from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.types import AttributeValueT, CompressionT, ExampleT


class ShardWriterBase(ABC):
    """Base class for shard writing. Buffers examples in memory so the shard
    is pickleable and format-specific serialization in `close()` can run in a
    different process.
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
        self._examples: list[ExampleT] = []

        # Make sure that the compression is supported for this shard file type.
        assert dataset_structure.compression in self.supported_compressions()

    @staticmethod
    def attribute_value_to_bytes(
        attribute: Attribute,
        value: AttributeValueT,
    ) -> bytes:
        """Validate and convert a `"bytes"` attribute value into `bytes`.

        Supports `bytes`, `bytearray`, `memoryview`, or an integer
        array/sequence with values in `[0, 255]` (e.g., `uint8`, `int32`,
        `int64` saved as raw byte values without extra zero bytes).
        """
        if isinstance(value, (bytes, bytearray, memoryview)):
            if not attribute.has_variable_size():
                current_shape = (len(value),)
                if current_shape != attribute.shape:
                    raise ValueError(
                        f"Attribute {attribute.name} has shape "
                        f"{current_shape} expected {attribute.shape}")
            return bytes(value)

        if isinstance(value, (str, bool, int, float, np.number)):
            raise ValueError(
                f"Cannot convert scalar {type(value)} to bytes for "
                f"{attribute = }")

        value_np = np.asarray(value)
        if not attribute.has_variable_size():
            if value_np.shape != attribute.shape:
                raise ValueError(f"Attribute {attribute.name} has shape "
                                 f"{value_np.shape} expected {attribute.shape}")

        if (value_np.ndim == 0 or
                not np.issubdtype(value_np.dtype, np.integer)):
            raise ValueError("Cannot cast value of dtype "
                             f"{value_np.dtype} passed as "
                             f"{attribute = }")

        if value_np.size > 0 and (int(value_np.min()) < 0 or
                                  int(value_np.max()) > 255):
            raise ValueError("Byte values must be in [0, 255] for "
                             f"{attribute = }, got range "
                             f"[{int(value_np.min())}, "
                             f"{int(value_np.max())}]")

        return np.asarray(value_np, dtype=np.uint8, order="C").tobytes()

    def write(self, values: ExampleT) -> None:
        """Buffer an example in memory to be serialized on `close()`.

        Args:

            values (ExampleT): Attribute values.
        """
        example: ExampleT = {}
        # Check the values are correct type and shape.
        for attribute in self.dataset_structure.saved_data_description:
            value = values[attribute.name]
            if attribute.dtype == "bytes":
                example[attribute.name] = self.attribute_value_to_bytes(
                    attribute=attribute,
                    value=value,
                )
                continue

            if attribute.dtype == "str" and not isinstance(value, str):
                raise ValueError(
                    f"Expected str for attribute {attribute.name}, got "
                    f"{type(value)}")

            # If the attribute dtype is "bytes" and shape is empty tuple we
            # consider this a variable size attribute and do not check shape.
            if not attribute.has_variable_size():
                # Else check the shape (the value should be a NumPy array but
                # maybe it is an int or bytearray).
                current_shape = np.asarray(value).shape
                if current_shape != attribute.shape:
                    raise ValueError(
                        f"Attribute {attribute.name} has shape "
                        f"{current_shape} expected {attribute.shape}")

            example[attribute.name] = (np.copy(value) if isinstance(
                value, np.ndarray) else value)

        self._examples.append(example)

    def close(self) -> tuple[str, ...]:
        """Serialize buffered examples into the shard file(-s) and return hash
        checksums.
        """
        if not self._examples:
            # Nothing to save.
            assert not self._shard_file.is_file()
            return ()

        # Make sure that the directory exists before writing the shard file.
        self._shard_file.parent.mkdir(exist_ok=True, parents=True)

        hash_checksums = self._close()
        self._examples.clear()
        assert self._shard_file.is_file()
        return hash_checksums

    @abstractmethod
    def _close(self) -> tuple[str, ...]:
        """Implementation of shard file(-s) serialization for non-empty
        `self._examples`.
        """

    @staticmethod
    @abstractmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """

    def _compute_file_hash_checksums(self) -> tuple[str, ...]:
        """Compute hash checksums of the shard file(-s). """
        return sedpack.io.utils.hash_checksums(
            file_path=self._shard_file,
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )
