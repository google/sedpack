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
from numpy import typing as npt

from sedpack.io.metadata import Attribute, DatasetStructure
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

        # A prefix such that prepended it creates a new name without collision
        # with any attribute name.
        self._counting_prefix: str = "len" + "_" * max(
            len(attribute.name)
            for attribute in dataset_structure.saved_data_description)

    def _value_to_np(
        self,
        attribute: Attribute,
        value: AttributeValueT,
    ) -> npt.NDArray[np.generic] | str:
        match attribute.dtype:
            case "bytes":
                raise ValueError("Attributes bytes are saved extra")
            case "str":
                assert isinstance(value, str)
                return value
            case _:
                return np.copy(value)

    def _write(self, values: ExampleT) -> None:
        """Write an example on disk. Writing may be buffered.

        Args:

            values (dict[str, npt.NDArray[np.generic]]): Attribute values.
        """
        # Just buffer all values.
        if not self._buffer:
            self._buffer = {}

        for (name, value), attribute in zip(
                values.items(),
                self.dataset_structure.saved_data_description,
        ):
            if attribute.dtype != "bytes":
                current_values = self._buffer.get(name, [])
                current_values.append(
                    self._value_to_np(
                        attribute=attribute,
                        value=value,
                    ))
                self._buffer[name] = current_values
            else:
                # Extend and remember the length.  Attributes with dtype "bytes"
                # may have variable length. Handle this case. We need to avoid
                # two thigs:
                # - Having wrong length of the bytes array and ideally also
                # avoid padding.
                # - Using allow_pickle when saving since that could lead to code
                # execution when loading malicious dataset.
                # We prefix the attribute name by `len_?` such that the new name
                # is unique and tells us the lengths of the byte arrays.
                counts = self._buffer.get(self._counting_prefix + name, [0])
                counts.append(counts[-1] +
                              len(value)  # type: ignore[arg-type,operator]
                             )
                self._buffer[self._counting_prefix + name] = counts

                byte_list: list[list[int]]
                byte_list = self._buffer.get(  # type: ignore[assignment]
                    name, [[]])
                byte_list[0].extend(list(value)  # type: ignore[arg-type]
                                   )
                self._buffer[name] = byte_list  # type: ignore[assignment]

    def close(self) -> None:
        """Close the shard file(-s).
        """
        if not self._buffer:
            assert not self._shard_file.is_file()
            return

        # Deal properly with "bytes" attributes.
        for attribute in self.dataset_structure.saved_data_description:
            if attribute.dtype != "bytes":
                continue
            self._buffer[attribute.name] = [
                np.array(
                    self._buffer[attribute.name][0],
                    dtype=np.uint8,
                )
            ]

        # Write the buffer into a file.
        match self.dataset_structure.compression:
            case "ZIP":
                np.savez_compressed(
                    str(self._shard_file),
                    allow_pickle=False,
                    **self._buffer,  # type: ignore[arg-type]
                )
            case "":
                np.savez(
                    str(self._shard_file),
                    allow_pickle=False,
                    **self._buffer,  # type: ignore[arg-type]
                )
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
