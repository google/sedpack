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

from pathlib import Path
import pickle
from typing import get_args

import numpy as np
import pytest

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.shard.get_shard_writer import (
    _SHARD_FILE_TYPE_TO_CLASS,
    get_shard_writer,
)
from sedpack.io.types import CompressionT, ShardFileTypeT
from sedpack.io.utils import is_module_present


def test_all_file_types_supported() -> None:
    assert set(_SHARD_FILE_TYPE_TO_CLASS.keys()) == set(
        get_args(ShardFileTypeT))


@pytest.mark.parametrize(
    "shard_file_type,compression",
    [
        ("fb", "LZ4"),
        ("npz", "ZIP"),
        ("tfrec", "GZIP"),
    ],
)
def test_shard_writer_is_pickleable_and_buffers_examples(
    tmp_path: Path,
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
) -> None:
    if shard_file_type == "tfrec" and not is_module_present("tensorflow"):
        pytest.skip("TensorFlow is optional, skip test if not present.")

    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(name="a", shape=(4,), dtype="float32"),
        ],
        shard_file_type=shard_file_type,
        compression=compression,
    )
    shard_file = tmp_path / f"shard.{shard_file_type}"
    writer = get_shard_writer(
        dataset_structure=dataset_structure,
        shard_file=shard_file,
    )

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    writer.write({"a": arr})
    # Mutating the input array after write must not mutate the buffered example.
    arr[0] = 999.0
    writer.write({"a": arr})

    assert len(writer._examples) == 2
    assert not shard_file.is_file()

    # Round-trip through pickle (simulating sending to another process) and
    # close the unpickled writer.
    unpickled_writer = pickle.loads(pickle.dumps(writer))
    checksums = unpickled_writer.close()

    assert shard_file.is_file()
    assert len(checksums) == len(dataset_structure.hash_checksum_algorithms)
    assert unpickled_writer._examples == []


@pytest.mark.parametrize(
    "shard_file_type,compression",
    [
        ("fb", "LZ4"),
        ("npz", "ZIP"),
        ("tfrec", "GZIP"),
    ],
)
def test_close_empty_shard(
    tmp_path: Path,
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
) -> None:
    if shard_file_type == "tfrec" and not is_module_present("tensorflow"):
        pytest.skip("TensorFlow is optional, skip test if not present.")

    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(name="a", shape=(4,), dtype="float32"),
        ],
        shard_file_type=shard_file_type,
        compression=compression,
    )
    shard_file = tmp_path / f"empty_shard.{shard_file_type}"
    writer = get_shard_writer(
        dataset_structure=dataset_structure,
        shard_file=shard_file,
    )

    assert writer.close() == ()
    assert not shard_file.is_file()
