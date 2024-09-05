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

import asyncio
from pathlib import Path
from typing import get_args
import pytest

import numpy as np

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.types import ShardFileTypeT

from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.npz import IterateShardNP
from sedpack.io.shard.get_shard_writer import get_shard_writer, _SHARD_FILE_TYPE_TO_CLASS

pytest_plugins = ("pytest_asyncio",)


async def shard_write_and_read(attributes: dict[str, np.ndarray],
                               shard_file: Path,
                               shard_file_type: ShardFileTypeT) -> None:
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(
                name=name,
                shape=value.shape[1:],
                dtype=str(value.dtype),
            ) for name, value in attributes.items()
        ],
        shard_file_type=shard_file_type,
        compression="",
    )

    # Write data into the file.
    writer = get_shard_writer(dataset_structure=dataset_structure,
                              shard_file=shard_file)
    one_value = next(iter(attributes.values()))  # One of the values.
    for i in range(one_value.shape[0]):
        writer.write(values={
            name: value[i] for name, value in attributes.items()
        })
    writer.close()

    iterate_shard: IterateShardBase
    match shard_file_type:
        case "npz":
            iterate_shard = IterateShardNP(dataset_structure=dataset_structure,
                                           process_record=None)
        case "fb":
            iterate_shard = IterateShardFlatBuffer(
                dataset_structure=dataset_structure, process_record=None)
        case _:
            raise ValueError(f"Unknown {shard_file_type = }")

    # Read those back.
    seen: int = 0
    i: int = 0
    async for example in iterate_shard.iterate_shard_async(shard_file):
        for name, value in attributes.items():
            np.testing.assert_allclose(example[name], value[i])
        seen += 1
        i += 1  # manual enumerate
    assert seen == one_value.shape[0]


@pytest.mark.asyncio
async def test_async_npz_with_int(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="npz")


@pytest.mark.asyncio
async def test_async_npz_with_float(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="npz")


@pytest.mark.asyncio
async def test_async_npz_mixed(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="npz")


@pytest.mark.asyncio
async def test_async_fb_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="fb")


@pytest.mark.asyncio
async def test_async_fb_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="fb")


@pytest.mark.asyncio
async def test_async_fb_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    await shard_write_and_read(attributes, shard_file, shard_file_type="fb")
