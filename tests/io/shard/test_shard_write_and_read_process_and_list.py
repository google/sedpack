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
from typing import get_args

import numpy as np

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.types import ShardFileTypeT

from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.npz import IterateShardNP
from sedpack.io.tfrec import IterateShardTFRec
from sedpack.io.shard.get_shard_writer import get_shard_writer, _SHARD_FILE_TYPE_TO_CLASS


def shard_write_and_read(attributes: dict[str, np.ndarray], shard_file: Path,
                         shard_file_type: ShardFileTypeT,
                         process_record) -> None:
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
        case "tfrec":
            iterate_shard = IterateShardTFRec(
                dataset_structure=dataset_structure,
                process_record=process_record)
        case "npz":
            iterate_shard = IterateShardNP(dataset_structure=dataset_structure,
                                           process_record=process_record)
        case "fb":
            iterate_shard = IterateShardFlatBuffer(
                dataset_structure=dataset_structure,
                process_record=process_record)
        case _:
            raise ValueError(f"Unknown {shard_file_type = }")

    # Read those back.
    seen: int = 0
    to_add = 1 if process_record else 0
    for i, example in enumerate(iterate_shard.process_and_list(shard_file)):
        for name, value in attributes.items():
            np.testing.assert_allclose(example[name], value[i] + to_add)
        seen += 1
    assert seen == one_value.shape[0]


def test_process_and_list_npz_with_int(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=None)


def test_process_and_list_npz_with_float(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=None)


def test_process_and_list_npz_mixed(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=None)


def test_process_and_list_tfrec_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=None)


def test_process_and_list_tfrec_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array(np.random.uniform(size=(10, 15)), dtype=np.float32),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=None)


def test_process_and_list_tfrec_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array(np.random.uniform(size=(10, 15)), dtype=np.float32),
        "b": np.array(np.random.uniform(size=(10, 25)), dtype=np.float32),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=None)


def test_process_and_list_fb_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=None)


def test_process_and_list_fb_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=None)


def test_process_and_list_fb_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=None)


def test_processing_process_and_list_npz_with_int(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_npz_with_float(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_npz_mixed(tmp_path):
    shard_file = tmp_path / "shard_file.npz"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="npz",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_tfrec_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_tfrec_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array(np.random.uniform(size=(10, 15)), dtype=np.float32),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_tfrec_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array(np.random.uniform(size=(10, 15)), dtype=np.float32),
        "b": np.array(np.random.uniform(size=(10, 25)), dtype=np.float32),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="tfrec",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_fb_with_int(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.array([[13 + 512, 2, 3], [4, 5, 6]]),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_fb_with_float(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})


def test_processing_process_and_list_fb_mixed(tmp_path):
    shard_file = tmp_path / "shard_file"
    attributes = {
        "a": np.random.uniform(size=(10, 15)),
        "b": np.random.uniform(size=(10, 25)),
        "c": np.random.randint(-5, 20, size=(10, 21)),
    }
    shard_write_and_read(attributes,
                         shard_file,
                         shard_file_type="fb",
                         process_record=lambda x: {name: value + 1 for name, value in x.items()})
