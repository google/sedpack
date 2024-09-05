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

import os
from pathlib import Path
import random
from typing import Any, Union

import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata, DatasetStructure, Attribute
from sedpack.io.types import CompressionT, ShardFileTypeT


def get_dataset(tmpdir: Union[str, Path]) -> Dataset:
    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="float32",
            shape=(138,),
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,)

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    return dataset


def fill(dataset, split, data):
    # Fill data in the dataset
    with dataset.filler() as filler:
        for attribute_value in data:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=split,
            )

    # Check the data is correct
    # Reopen the dataset
    dataset = Dataset(dataset.path)
    dataset.check()


def check_presence(dataset, split, data):
    for i, example in enumerate(
            dataset.as_numpy_iterator(
                split=split,
                shuffle=0,
                repeat=False,
            )):
        assert np.allclose(example["attribute_name"], data[i:i + 1])

    # We tested everything
    assert i + 1 == data.shape[0], "Not all examples have been iterated"


def test_continue_writing_another_split(tmpdir: Union[str, Path]) -> None:
    """Check that we can write more examples into empty / single split. This
    would uncover the bug addressed by merging updates info.
    """
    data_train = np.random.random((1024, 138))
    filled_train: int = 0
    data_test = np.random.random((1024, 138))
    filled_test: int = 0

    dataset = get_dataset(tmpdir)

    fill_now: int = 20

    for _ in range(4):
        fill_now = random.randint(10, 50)
        fill(
            dataset=dataset,
            split="train",
            data=data_train[filled_train:filled_train + fill_now],
        )
        filled_train += fill_now

        fill_now = random.randint(10, 50)
        fill(
            dataset=dataset,
            split="test",
            data=data_test[filled_test:filled_test + fill_now],
        )
        filled_test += fill_now

        # Both splits are present after writing (not just directly after
        # writing into one).
        check_presence(dataset, "train", data_train[:filled_train])
        check_presence(Dataset(dataset.path), "train",
                       data_train[:filled_train])
        assert dataset._dataset_info.splits[
            "train"].number_of_examples == filled_train
        check_presence(dataset, "test", data_test[:filled_test])
        check_presence(Dataset(dataset.path), "test", data_test[:filled_test])
        assert dataset._dataset_info.splits[
            "test"].number_of_examples == filled_test


def test_local_root_path(tmpdir: Union[str, Path]) -> None:
    """Check that relative path checks work even when dataset root path is
    local. For this we need to write multiple times in the dataset.
    """
    # Change the working directory to be in the /tmp/pytest-of-user/
    os.chdir(tmpdir)

    data_train = np.random.random((1024, 138))
    filled_train: int = 0
    data_test = np.random.random((1024, 138))
    filled_test: int = 0

    dataset = get_dataset("my_dataset")

    fill_now: int = 20

    for _ in range(4):
        fill_now = random.randint(10, 50)
        fill(
            dataset=dataset,
            split="train",
            data=data_train[filled_train:filled_train + fill_now],
        )
        filled_train += fill_now

        fill_now = random.randint(10, 50)
        fill(
            dataset=dataset,
            split="test",
            data=data_test[filled_test:filled_test + fill_now],
        )
        filled_test += fill_now
