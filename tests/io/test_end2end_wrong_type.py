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
import pytest
from typing import Any, Union

import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata, DatasetStructure, Attribute
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT


def end2end(tmpdir: Union[str, Path], input_dtype: npt.DTypeLike, saved_dtype: npt.DTypeLike, method: str,
            shard_file_type: ShardFileTypeT, compression: CompressionT) -> None:
    array_of_values = np.random.random((10, 138)) * 256
    array_of_values = np.array(array_of_values, dtype=input_dtype)
    #print(f">>> {array_of_values.dtype = }")
    #print(f">>> {array_of_values.shape = }")
    #print(f">>> {array_of_values[0, :10] = }")
    #print(f">>> {array_of_values.tobytes() = }")

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="filling_before",
            dtype=str(array_of_values.dtype),
            shape=array_of_values.shape,
        ),
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=str(saved_dtype),
            shape=array_of_values[0].shape,
        ),
        sedpack.io.metadata.Attribute(
            name="filling_after",
            dtype=str(array_of_values.dtype),
            shape=array_of_values.shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression=compression,
        examples_per_shard=256,
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
                values={
                    "filling_before": array_of_values,
                    "attribute_name": attribute_value,
                    "filling_after": array_of_values,
                },
                split=TRAIN_SPLIT,
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
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i:i + 1])
        case "as_numpy_iterator":
            for i, example in enumerate(
                    dataset.as_numpy_iterator(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i])
        case "as_numpy_iterator_concurrent":
            for i, example in enumerate(
                    dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert np.allclose(example["attribute_name"],
                                   array_of_values[i])

    # We tested everything
    assert i + 1 == array_of_values.shape[
        0], "Not all examples have been iterated"


def test_end2end_wrong_value_type(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            input_dtype="uint8",
            saved_dtype="int32",
            method="as_numpy_iterator",
            shard_file_type="fb",
            compression="LZ4")


def test_end2end_wrong_value_type_no_cast(
        tmpdir: Union[str, Path]) -> None:
    with pytest.raises(ValueError) as e:
        end2end(tmpdir=tmpdir,
                input_dtype="float32",
                saved_dtype="uint8",
                method="as_numpy_iterator",
                shard_file_type="fb",
                compression="LZ4")
