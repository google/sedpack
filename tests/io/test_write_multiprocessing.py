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
from typing import Any, Union

import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset, DatasetFiller
from sedpack.io import Metadata, DatasetStructure, Attribute
from sedpack.io.types import SplitT, TRAIN_SPLIT


def feed_writer(dataset_filler: DatasetFiller,
                array_of_values: npt.NDArray[np.generic], split: SplitT) -> int:
    # Fill data in the dataset

    with dataset_filler as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=split,
            )

    return len(array_of_values)


def test_write_multiprocessing(tmpdir: Union[str, Path]) -> None:
    dtype = "float32"
    array_of_values = np.random.random((1024, 138))
    array_of_values = array_of_values.astype(dtype)

    tiny_experiment_path: Path = Path(tmpdir) / "write_multiprocessing"

    # Create a dataset

    dataset_metadata = Metadata(
        description="Test of the lib write_multiprocessing")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=str(dtype),
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="GZIP",
        examples_per_shard=256,
    )

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    custom_arguments = [
        (array_of_values[:100],),
        (array_of_values[100:900],),
        (array_of_values[900:],),
    ]
    custom_kwarguments = [
        {
            "split": "train"
        },
        {
            "split": "train"
        },
        {
            "split": "train"
        },
    ]

    results = dataset.write_multiprocessing(feed_writer,
                                            custom_arguments,
                                            custom_kwarguments,
                                            single_process=True)

    assert results == [
        len(part_of_array_of_values[0])
        for part_of_array_of_values in custom_arguments
    ]

    # Check the data is correct

    for i, example in enumerate(
            dataset.as_tfdataset(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=1,
            )):
        assert np.allclose(example["attribute_name"], array_of_values[i:i + 1])

    # We tested everything
    assert i + 1 == array_of_values.shape[
        0], "Not all examples have been iterated"
