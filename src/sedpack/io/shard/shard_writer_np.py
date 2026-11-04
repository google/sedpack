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
"""Dataset shard manipulation.

For information how to read and write TFRecord files see
https://www.tensorflow.org/tutorials/load_data/tfrecord
"""

from pathlib import Path

import numpy as np
from numpy import typing as npt

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.shard.shard_writer_base import ShardWriterBase
from sedpack.io.types import AttributeValueT, CompressionT


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
                return np.asarray(value)

    def _close(self) -> tuple[str, ...]:
        """Serialize buffered examples into an NPZ shard file.
        """
        buffer: dict[str, list[AttributeValueT]] = {}
        for attribute in self.dataset_structure.saved_data_description:
            name = attribute.name
            if attribute.dtype != "bytes":
                buffer[name] = [
                    self._value_to_np(
                        attribute=attribute,
                        value=example[name],
                    ) for example in self._examples
                ]
            else:
                # Extend and remember the length. Attributes with dtype "bytes"
                # may have variable length. Handle this case. We need to avoid
                # two things:
                # - Having wrong length of the bytes array and ideally also
                # avoid padding.
                # - Using allow_pickle when saving since that could lead to code
                # execution when loading malicious dataset.
                # We prefix the attribute name by `len_?` such that the new name
                # is unique and tells us the lengths of the byte arrays.
                byte_chunks: list[bytes] = [
                    self.attribute_value_to_bytes(
                        attribute=attribute,
                        value=example[name],
                    ) for example in self._examples
                ]
                counts: list[int] = [0]
                for chunk in byte_chunks:
                    counts.append(counts[-1] + len(chunk))
                buffer[self._counting_prefix +
                       name] = counts  # type: ignore[assignment]
                buffer[name] = [
                    np.frombuffer(b"".join(byte_chunks), dtype=np.uint8)
                ]

        # Write the buffer into a file. We should not need to allow_pickle while
        # saving (the default value is True). But on GitHub actions macos-13
        # runner the tests were failing while reading. The security concern
        # (code execution) should be more on the side of loading pickled data.
        match self.dataset_structure.compression:
            case "ZIP":
                np.savez_compressed(
                    str(self._shard_file),
                    **buffer,  # type: ignore[arg-type]
                )
            case "":
                np.savez(
                    str(self._shard_file),
                    **buffer,  # type: ignore[arg-type]
                )
            case _:
                # Default should never happen since ShardWriterBase checks that
                # the requested compression type is supported.
                raise ValueError(f"Unsupported compression type "
                                 f"{self.dataset_structure.compression} in "
                                 f"ShardWriterNP, supported values are: "
                                 f"{self.supported_compressions()}")

        return self._compute_file_hash_checksums()

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return ["ZIP", ""]
