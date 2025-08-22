# Copyright 2025 Google LLC
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

from pathlib import Path
import pytest
from typing import Union

import numpy as np

import sedpack
from sedpack.io.iteration import RustGenerator
from sedpack.io.metadata import DatasetStructure


def test_wrong_file_paralelism() -> None:
    with pytest.raises(
            ValueError,
            match="The argument file_parallelism should be positive.*",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(),
            shard_iterator=[],
            process_record=None,
            file_parallelism=0,
        )


def test_wrong_shard_type() -> None:
    with pytest.raises(
            ValueError,
            match="RustGenerator is implemented only for FlatBuffers.",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(shard_file_type="tfrec"),
            shard_iterator=[],
            process_record=None,
            file_parallelism=1,
        )


def test_wrong_compression() -> None:
    with pytest.raises(
            ValueError,
            match=
            "The compression .* is not among the supported compressions: .*",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(
                shard_file_type="fb",
                compression="ZIP",
            ),
            shard_iterator=[],
            process_record=None,
            file_parallelism=1,
        )
