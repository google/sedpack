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
from typing import Any

import tensorflow as tf

from sedpack.io.metadata import DatasetStructure
from sedpack.io.tfrec.tfdata import to_tfrecord
from sedpack.io.types import ExampleT, CompressionT
from sedpack.io.shard.shard_writer_base import ShardWriterBase


class ShardWriterTFRec(ShardWriterBase):
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
        assert dataset_structure.shard_file_type == "tfrec"

        super().__init__(
            dataset_structure=dataset_structure,
            shard_file=shard_file,
        )

        # Open the tf.io.TFRecordWriter only with the first `write` call. Make
        # it None immediately during a call to `close`.
        self._tf_shard_writer: Any | None = None

    def _write(self, values: ExampleT) -> None:
        """Write an example on disk. Writing may be buffered.

        Args:

            values (ExampleT): Attribute values.
        """
        if (self.dataset_structure.compression
                not in ShardWriterTFRec.supported_compressions()):
            raise ValueError(
                f"Unsupported compression {self.dataset_structure.compression}"
                " requested for TFRecordWriter, expected "
                f"{ShardWriterTFRec.supported_compressions()}")
        if not self._tf_shard_writer:
            self._tf_shard_writer = tf.io.TFRecordWriter(
                str(self._shard_file),
                self.dataset_structure.compression,  # type: ignore
            )

        example = to_tfrecord(
            saved_data_description=self.dataset_structure.
            saved_data_description,
            values=values,
        )
        self._tf_shard_writer.write(example)

    def close(self) -> None:
        """Close the shard file(-s).
        """
        if not self._tf_shard_writer:
            raise ValueError("Trying to close a shard that was not open")
        self._tf_shard_writer.close()
        self._tf_shard_writer = None

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return ["GZIP", "ZLIB", ""]
