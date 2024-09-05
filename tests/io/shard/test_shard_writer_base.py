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

from typing import get_args

from sedpack.io.types import ShardFileTypeT
from sedpack.io.shard.shard_writer_base import ShardWriterBase
from sedpack.io.shard.get_shard_writer import get_shard_writer, _SHARD_FILE_TYPE_TO_CLASS


def test_all_file_types_supported():
    assert set(_SHARD_FILE_TYPE_TO_CLASS.keys()) == set(
        get_args(ShardFileTypeT))
