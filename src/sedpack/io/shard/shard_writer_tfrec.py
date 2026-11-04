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

from sedpack.io.metadata import DatasetStructure
from sedpack.io.tfrec.tfdata import to_tfrecord
from sedpack.io.types import CompressionT
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

    def _close(self) -> tuple[str, ...]:
        """Serialize buffered examples into a TFRecord shard file.
        """
        # TensorFlow is an optional dependency.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        with tf.io.TFRecordWriter(
                str(self._shard_file),
                self.dataset_structure.compression,  # type: ignore[arg-type]
        ) as writer:
            for example in self._examples:
                writer.write(
                    to_tfrecord(
                        saved_data_description=self.dataset_structure.
                        saved_data_description,
                        values=example,
                    ))

        return self._compute_file_hash_checksums()

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return ["GZIP", "ZLIB", ""]
