# Copyright 2023-2024 Google LLC
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
from typing import Union

import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT


def end2end(tmpdir: Union[str, Path], dtype: npt.DTypeLike, method: str,
            shard_file_type: ShardFileTypeT, compression: CompressionT) -> None:
    examples_per_shard: int = 256
    array_of_values = np.random.random((examples_per_shard * 5, 138))
    array_of_values = array_of_values.astype(dtype)
    custom_metadata_type_limit: int = array_of_values.shape[
        0] // examples_per_shard

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=str(dtype),
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression=compression,
        examples_per_shard=examples_per_shard,
        shard_file_type=shard_file_type,
    )

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset

    with dataset.filler() as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
                custom_metadata={
                    "take": "me",
                    "some_number": 0
                },
            )

        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
                custom_metadata={
                    "some_number": 0,
                    "take": "me"
                },  # same, so ignored
            )

    # Check the data is correct
    # Reopen the dataset
    dataset = Dataset(tiny_experiment_path)
    dataset.check()

    match method:
        case "as_tfdataset":
            for i, example in enumerate(
                    dataset.as_tfdataset(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                        batch_size=1,
                        custom_metadata_type_limit=custom_metadata_type_limit,
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i:i + 1])
        case "as_numpy_iterator":
            for i, example in enumerate(
                    dataset.as_numpy_iterator(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                        custom_metadata_type_limit=custom_metadata_type_limit,
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i])
        case "as_numpy_iterator_concurrent":
            for i, example in enumerate(
                    dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                        custom_metadata_type_limit=custom_metadata_type_limit,
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i])

    # We tested everything
    assert i + 1 == array_of_values.shape[
        0], "Not all examples have been iterated"


def test_end2end_as_tfdataset_tfrec_float32_float32(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_tfdataset",
            shard_file_type="tfrec",
            compression="GZIP")


def test_end2end_as_numpy_iterator_concurrent_tfrec(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_concurrent",
            shard_file_type="tfrec",
            compression="GZIP")


def test_end2end_as_numpy_iterator_tfrec(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator",
            shard_file_type="tfrec",
            compression="GZIP")


def test_end2end_as_numpy_iterator_npz(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator",
            shard_file_type="npz",
            compression="ZIP")


def test_end2end_as_numpy_iterator_concurrent_fb(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_concurrent",
            shard_file_type="fb",
            compression="LZ4")


def test_end2end_as_numpy_iterator_fb(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator",
            shard_file_type="fb",
            compression="LZ4")
