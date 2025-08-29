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
"""Iterate a npz shard. See sedpack.io.shard.shard_writer_np.ShardWriterNP
for more information how a npz shard is saved.
"""

import io
from pathlib import Path
from typing import AsyncIterator, Iterable

import aiofiles
import numpy as np

from sedpack.io.metadata import Attribute
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.types import AttributeValueT, ExampleT
from sedpack.io.utils import func_or_identity


class IterateShardNP(IterateShardBase[T]):
    """Iterate a shard saved in the npz format.
    """

    @staticmethod
    def decode_attribute(
        np_value: AttributeValueT,
        attribute: Attribute,
    ) -> AttributeValueT:
        match attribute.dtype:
            case "str":
                return str(np_value)
            case "bytes":
                raise ValueError("One needs to use decode_bytes_attribute")
            case "int":
                return int(np_value)
            case _:
                return np_value

    @staticmethod
    def decode_bytes_attribute(
        value: AttributeValueT,
        indexes: list[AttributeValueT],
        attribute: Attribute,
        index: int,
    ) -> AttributeValueT:
        """Decode a bytes attribute. We are saving the byte attributes as a
        continuous array across multiple examples and on the side we also save
        the indexes into this array.

        Args:

          value (AttributeValueT): The NumPy array of np.uint8 containing
          concatenated bytes values.

          indexes (list[AttributeValueT]): Indexes into this array.

          attribute (Attribute): The attribute description.

          index (int): Which example out of this shard to return.
        """
        if attribute.dtype != "bytes":
            raise ValueError("One needs to use decode_attribute")
        # Help with type-checking:
        my_value = np.array(value, np.uint8)
        my_indexes = np.array(indexes, np.int64)
        del value
        del indexes

        begin: int = my_indexes[index]
        end: int = my_indexes[index + 1]
        return bytes(my_value[begin:end])

    def iterate_shard(self, file_path: Path) -> Iterable[ExampleT]:
        """Iterate a shard saved in the NumPy format npz.
        """
        # A prefix such that prepended it creates a new name without collision
        # with any attribute name.
        self._counting_prefix: str = "len" + "_" * max(
            len(attribute.name)
            for attribute in self.dataset_structure.saved_data_description)

        shard_content: dict[str, list[AttributeValueT]] = np.load(
            file_path,
            allow_pickle=False,
        )

        # A given shard contains the same number of elements for each
        # attribute.
        elements: int
        first_attribute = self.dataset_structure.saved_data_description[0]
        if self._counting_prefix + first_attribute.name in shard_content:
            elements = len(
                shard_content[self._counting_prefix + first_attribute.name]) - 1
        else:
            elements = len(shard_content[first_attribute.name])

        for i in range(elements):
            yield {
                attribute.name:
                    IterateShardNP.decode_attribute(
                        np_value=shard_content[attribute.name][i],
                        attribute=attribute,
                    ) if attribute.dtype != "bytes" else
                    IterateShardNP.decode_bytes_attribute(
                        value=shard_content[attribute.name][0],
                        indexes=shard_content[self._counting_prefix +
                                              attribute.name],
                        attribute=attribute,
                        index=i,
                    )
                for attribute in self.dataset_structure.saved_data_description
            }

    # TODO(issue #85) fix and test async iterator typing
    async def iterate_shard_async(  # pylint: disable=invalid-overridden-method
        self,
        file_path: Path,
    ) -> AsyncIterator[ExampleT]:
        """Asynchronously iterate a shard saved in the NumPy format npz.
        """
        async with aiofiles.open(file_path, "rb") as f:
            content_bytes: bytes = await f.read()
            content_io = io.BytesIO(content_bytes)

        shard_content: dict[str, list[AttributeValueT]] = np.load(
            content_io,
            allow_pickle=False,
        )

        # A given shard contains the same number of elements for each
        # attribute.
        elements: int = 0
        for values in shard_content.values():
            elements = len(values)
            break

        for i in range(elements):
            yield {name: value[i] for name, value in shard_content.items()}

    def process_and_list(self, shard_file: Path) -> list[T]:
        process_record = func_or_identity(self.process_record)

        return [
            process_record(example)
            for example in self.iterate_shard(file_path=shard_file)
        ]
