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

from pathlib import Path

from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import ShardFileTypeT

from sedpack.io.shard.shard_writer_base import ShardWriterBase
from sedpack.io.shard.shard_writer_flatbuffer import ShardWriterFlatBuffer
from sedpack.io.shard.shard_writer_np import ShardWriterNP
from sedpack.io.shard.shard_writer_tfrec import ShardWriterTFRec

_SHARD_FILE_TYPE_TO_CLASS: dict[ShardFileTypeT, type[ShardWriterBase]] = {
    "tfrec": ShardWriterTFRec,
    "npz": ShardWriterNP,
    "fb": ShardWriterFlatBuffer,
}


def get_shard_writer(dataset_structure: DatasetStructure,
                     shard_file: Path) -> ShardWriterBase:
    """Return the right subclass of ShardWriterBase.
    """
    return _SHARD_FILE_TYPE_TO_CLASS[dataset_structure.shard_file_type](
        dataset_structure=dataset_structure,
        shard_file=shard_file,
    )
