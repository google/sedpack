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
"""Test all shard file types with as_tfdataset.
"""

import itertools
from pathlib import Path
from typing import Callable, Union

import pytest
import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata
from sedpack.io.shard.shard_writer_flatbuffer import ShardWriterFlatBuffer
from sedpack.io.shard.shard_writer_np import ShardWriterNP
from sedpack.io.shard.shard_writer_tfrec import ShardWriterTFRec
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT

from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.types import ExampleT


def end2end(
    tmpdir: Union[str, Path],
    dtype: npt.DTypeLike,
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
    process_record: Callable[[ExampleT], T] | None,
) -> None:
    array_of_values = np.random.random((1024, 138))
    array_of_values = array_of_values.astype(dtype)

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
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    # Check the data is correct
    # Reopen the dataset
    dataset = Dataset(tiny_experiment_path)
    dataset.check()

    for i, example in enumerate(
            dataset.as_tfdataset(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=1,
                process_record=process_record,
            )):
        if process_record:
            assert np.allclose(
                example["attribute_name"],
                process_record({"attribute_name": array_of_values[i:i + 1]
                               })["attribute_name"],
            )
        else:
            assert np.allclose(example["attribute_name"],
                               array_of_values[i:i + 1])

    # We tested everything
    assert i + 1 == array_of_values.shape[
        0], "Not all examples have been iterated"


@pytest.mark.parametrize(
    "shard_file_type,compression,dtype,process_record",
    itertools.chain(
        itertools.product(
            ["tfrec"],
            ShardWriterTFRec.supported_compressions(),
            ["float16", "float32"],
            [
                None,
                lambda d: {
                    k: v + 1 for k, v in d.items()
                },
            ],
        ),
        itertools.product(
            ["npz"],
            ShardWriterNP.supported_compressions(),
            ["float16", "float32"],
            [
                None,
                lambda d: {
                    k: v + 1 for k, v in d.items()
                },
            ],
        ),
        itertools.product(
            ["fb"],
            ShardWriterFlatBuffer.supported_compressions(),
            ["float16", "float32"],
            [
                None,
                lambda d: {
                    k: v + 1 for k, v in d.items()
                },
            ],
        ),
    ),
)
def test_end2end_as_tfdataset(
    shard_file_type: str,
    compression: str,
    dtype: str,
    process_record: Callable[[ExampleT], T] | None,
    tmp_path: Union[str, Path],
) -> None:
    end2end(
        tmpdir=tmp_path,
        dtype=dtype,
        shard_file_type=shard_file_type,
        compression=compression,
        process_record=process_record,
    )
