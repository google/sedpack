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
import pytest
from typing import get_args

import numpy as np

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.shard.shard_writer_flatbuffer import ShardWriterFlatBuffer
from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.utils import import_sedpack_rs


# Skip all tests if we failed to import sedpack_rs.
sedpack_rs = import_sedpack_rs()
if sedpack_rs:
  from sedpack.io.flatbuffer.iterate_rs import IterateShardFlatBufferRs
pytestmark = pytest.mark.skipif(not sedpack_rs, reason="Need Rust binding")


def shard_write_and_read(attributes: dict[str, np.ndarray],
                         shard_file: Path) -> None:
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(
                name=name,
                shape=value.shape[1:],
                dtype=str(value.dtype),
            ) for name, value in attributes.items()
        ],
        shard_file_type="fb",
        compression="",
    )

    # Write data into the file.
    writer = ShardWriterFlatBuffer(dataset_structure=dataset_structure,
                                   shard_file=shard_file)
    one_value = next(iter(attributes.values()))  # One of the values.
    for i in range(one_value.shape[0]):
        writer.write(values={
            name: value[i] for name, value in attributes.items()
        })
    writer.close()

    iterate_shard: IterateShardBase
    iterate_shard = IterateShardFlatBufferRs(
        dataset_structure=dataset_structure, process_record=None)

    # Read those back.
    seen: int = 0
    for i, example in enumerate(iterate_shard.iterate_shard(shard_file)):
        for name, value in attributes.items():
            np.testing.assert_allclose(example[name], value[i])
        seen += 1
    assert seen == one_value.shape[0]


def test_fb_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes, shard_file)


def test_fb_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    shard_write_and_read(attributes, shard_file)


def test_fb_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes, shard_file)


def test_fb_all_dtypes(tmp_path):
    # Examples per shard
    E = 111
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a_int8": np.random.uniform(-5, 5, size=(E, 15)).astype(np.int8),
        "a_uint8": np.random.uniform(-5, 5, size=(E, 3, 15)).astype(np.uint8),
        "a_int16": np.random.uniform(-5, 5, size=(E, 15, 7, 3, 2)).astype(np.int16),
        "a_uint16": np.random.uniform(-5, 5, size=(E, 1, 1, 15)).astype(np.uint16),
        "a_int32": np.random.uniform(-5, 5, size=(E, 1)).astype(np.int32),
        "a_uint32": np.random.uniform(-5, 5, size=(E, 11, 15)).astype(np.uint32),
        "a_int64": np.random.uniform(-5, 5, size=(E, 15, 17, 19)).astype(np.int64),
        "a_uint64": np.random.uniform(-5, 5, size=(E, 2, 3, 15)).astype(np.uint64),
        "a_float16": np.random.uniform(-5, 5, size=(E, 15, 2, 3)).astype(np.float16),
        "a_float32": np.random.uniform(-5, 5, size=(E, 16)).astype(np.float32),
        "a_float64": np.random.uniform(-5, 5, size=(E, 1, 5)).astype(np.float64),
        "a_float128": np.random.uniform(-5, 5, size=(E, 5, 3)).astype(np.float128),
    }
    shard_write_and_read(attributes, shard_file)
