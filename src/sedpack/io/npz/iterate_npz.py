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
from typing import Iterable

import aiofiles
import numpy as np

from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.types import AttributeValueT, ExampleT
from sedpack.io.utils import func_or_identity


class IterateShardNP(IterateShardBase[T]):
    """Iterate a shard saved in the npz format.
    """

    def iterate_shard(self, file_path: Path) -> Iterable[ExampleT]:
        """Iterate a shard saved in the NumPy format npz.
        """
        shard_content: dict[str, list[AttributeValueT]] = np.load(file_path)

        # A given shard contains the same number of elements for each
        # attribute.
        elements: int = 0
        for values in shard_content.values():
            elements = len(values)
            break

        for i in range(elements):
            yield {name: value[i] for name, value in shard_content.items()}

    async def iterate_shard_async(self, file_path: Path):
        """Asynchronously iterate a shard saved in the NumPy format npz.
        """
        async with aiofiles.open(file_path, "rb") as f:
            content_bytes: bytes = await f.read()
            content_io = io.BytesIO(content_bytes)

        shard_content: dict[str, list[AttributeValueT]] = np.load(content_io)

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
