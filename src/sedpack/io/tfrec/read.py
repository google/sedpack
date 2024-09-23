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
"""Read a dataset using pure asyncio.
"""

import os
from pathlib import Path
from typing import Any, Callable, Iterable

import tensorflow as tf

from sedpack.io.metadata import DatasetStructure
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.tfrec.tfdata import get_from_tfrecord
from sedpack.io.types import ExampleT
from sedpack.io.utils import func_or_identity


class IterateShardTFRec(IterateShardBase[T]):
    """Iterate a TFRec shard.
    """

    def __init__(
        self,
        dataset_structure: DatasetStructure,
        process_record: Callable[[ExampleT], Any] | None,
        num_parallel_calls: int = os.cpu_count() or 4,
    ) -> None:
        super().__init__(dataset_structure=dataset_structure,
                         process_record=process_record)
        # This is not pickleable, but can be created on the fly.
        self.from_tfrecord: Callable[[Any], Any] | None = None
        self.num_parallel_calls: int = num_parallel_calls

    def iterate_shard(self, file_path: Path) -> Iterable[ExampleT]:
        """Iterate a shard saved in the TFRec format
        """
        if not self.from_tfrecord:
            self.from_tfrecord = get_from_tfrecord(
                self.dataset_structure.saved_data_description)

        # Read the shard.
        tf_dataset_records = tf.data.TFRecordDataset(
            str(file_path),
            compression_type=self.dataset_structure.compression,  # type: ignore
        )

        # Decode examples.
        tf_dataset_examples = tf_dataset_records.map(
            self.from_tfrecord,
            num_parallel_calls=self.num_parallel_calls,
        )

        yield from tf_dataset_examples.as_numpy_iterator()  # type: ignore

    async def iterate_shard_async(self, file_path: Path):
        raise NotImplementedError

    def process_and_list(self, shard_file: Path) -> list[T]:
        """Return a list of processed examples. Used as a function call in a
        different process. Returning a list as opposed to an iterator allows to
        do all work in another process and all that needs to be done is a
        memory copy between processes.

        TODO think of a way to avoid copying memory between processes.

        Args:

          shard_file (Path): Path to the shard file.

        Returns a list of examples present in the shard identified by the path
        where a `process_record` function has been applied (if not None).
        """
        process_record = func_or_identity(self.process_record)

        return [
            process_record(example)
            for example in self.iterate_shard(file_path=shard_file)
        ]
