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
"""Iterate a FlatBuffers shard. See src/sedpack/io/flatbuffer/shard.fbs
sedpack.io.shard.shard_writer_flatbuffer.ShardWriterFlatBuffer for more
information how it is saved.
"""

from pathlib import Path
from typing import Iterable, Callable

import numpy as np

# This class makes no sense without being able to import sedpack_rs.
import sedpack_rs  # type: ignore # pylint: disable=import-error

from sedpack.io.flatbuffer.iterate import IterateShardFlatBuffer
from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import ExampleT
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T


class IterateShardFlatBufferRs(IterateShardBase[T]):
    """Remember everything to be able to iterate shards. This can be pickled
    and passed as a callable object into another process.

    This class uses the Rust code present in sedpack_rs.
    """

    def __init__(self, dataset_structure: DatasetStructure,
                 process_record: Callable[[ExampleT], T] | None) -> None:
        if dataset_structure.compression != "":
            raise ValueError("No compression is currently supported by the "
                             "Rust binding")
        for attribute in dataset_structure.saved_data_description:
            if attribute.has_variable_size():
                raise ValueError("Only attributes with fixed shape are "
                                 f"supported by Rust binding, {attribute.name}"
                                 " does not have fixed shape")
        super().__init__(dataset_structure=dataset_structure,
                         process_record=process_record)

    def get_content(self, file_path: Path) -> dict[str, np.ndarray]:
        """Return whole shard content as a dictionary of attribute name and all
        attribute values in the shard. If there are `E` examples in the given
        shard the following holds:
        `result[attribute.name].shape = (E, *attribute.shape)`.
        """
        parsed: list[np.ndarray] = sedpack_rs.iterate_shard_py(str(file_path))
        result: dict[str, np.ndarray] = {}
        for np_bytes, attribute in zip(
                parsed, self.dataset_structure.saved_data_description):
            result[attribute.name] = IterateShardFlatBuffer.decode_array(
                np_bytes=np_bytes,
                attribute=attribute,
                batch_size=-1,
            )

        # Validity check.
        assert len(set(x.shape[0] for x in result.values())) <= 1

        return result

    def iterate_shard(self, file_path: Path) -> Iterable[ExampleT]:
        """Iterate a shard.
        """
        shard_content: dict[str, np.ndarray] = self.get_content(file_path)
        num_examples: int = 0
        try:
            num_examples = next(iter(shard_content.values())).shape[0]
        except StopIteration:  # Zero is fine.
            pass
        for i in range(num_examples):
            yield {name: value[i] for name, value in shard_content.items()}

    async def iterate_shard_async(self, file_path: Path):
        """Asynchronously iterate a shard.
        """
        raise NotImplementedError("TODO")

    def process_and_list(self, shard_file: Path) -> list[T]:
        """Return a list of processed examples. Used as a function call in a
        different process. Returning a list as opposed to an iterator allows to
        do all work in another process and all that needs to be done is a
        memory copy between processes.

        TODO think of a way to avoid copying memory between processes.

        Args:

            shard_file (Path): Path to the shard file.
        """
        raise NotImplementedError("TODO")
