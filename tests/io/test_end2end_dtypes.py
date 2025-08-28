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
from typing import Union

import numpy as np
import numpy.typing as npt
import pytest

import sedpack
from sedpack.io import Dataset
from sedpack.io.shard_info_iterator import ShardInfoIterator
from sedpack.io import Metadata
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT


def end2end_str(
    tmpdir: Union[str, Path],
    method: str,
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
) -> None:
    array_of_values = [
        "https://arxiv.org/abs/2306.07249",
        "Ḽơᶉëᶆ ȋṕšᶙṁ ḍỡḽǭᵳ ʂǐť ӓṁệẗ, ĉṓɲṩḙċťᶒțûɾ",
        "ấɖḯƥĭṩčįɳġ ḝłįʈ, șếᶑ ᶁⱺ ẽḭŭŝḿꝋď ṫĕᶆᶈṓɍ ỉñḉīḑȋᵭṵńť ṷŧ ḹẩḇőꝛế",
        "éȶ đꝍꞎôꝛȇ ᵯáꞡᶇā ąⱡîɋṹẵ.",
    ]

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_str_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="strange_strings",
            dtype="str",
            shape=(),
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression=compression,
        examples_per_shard=3,
        shard_file_type=shard_file_type,
    )

    # Test attribute_by_name
    for attribute in example_attributes:
        assert dataset_structure.attribute_by_name(
            attribute_name=attribute.name) == attribute

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset

    with dataset.filler() as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"strange_strings": attribute_value},
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
                # No idea how to have an actual string in TensorFlow. Maybe it
                # is best to leave it as a tensor anyway since that is the
                # "native" type.
                #assert type(example["strange_strings"][0]) == type(
                #    array_of_values[i])
                assert example["strange_strings"] == array_of_values[i:i + 1]
        case "as_numpy_iterator":
            for i, example in enumerate(
                    dataset.as_numpy_iterator(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert type(example["strange_strings"]) == type(
                    array_of_values[i])
                assert example["strange_strings"] == array_of_values[i]
        case "as_numpy_iterator_concurrent":
            for i, example in enumerate(
                    dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert type(example["strange_strings"]) == type(
                    array_of_values[i])
                assert example["strange_strings"] == array_of_values[i]

    # We tested everything
    assert i + 1 == len(array_of_values), "Not all examples have been iterated"

    # Number of shards matches
    full_iterator = ShardInfoIterator(
        dataset_path=dataset.path,
        dataset_info=dataset.dataset_info,
        split=None,
    )
    number_of_all_shards: int = full_iterator.number_of_shards()
    assert number_of_all_shards == len(full_iterator)
    assert number_of_all_shards == len(list(full_iterator))
    assert number_of_all_shards == sum(
        ShardInfoIterator(
            dataset_path=dataset.path,
            dataset_info=dataset.dataset_info,
            split=split,
        ).number_of_shards() for split in ["train", "test", "holdout"])


# TODO common fixture tfrec_dataset
@pytest.mark.parametrize("method", [
    "as_tfdataset",
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
])
def test_end2end_dtypes_str_tfrec(
    method: str,
    tmpdir: Union[str, Path],
) -> None:
    end2end_str(
        tmpdir=tmpdir,
        method=method,
        shard_file_type="tfrec",
        compression="GZIP",
    )


# TODO common fixture npz_dataset
@pytest.mark.parametrize("method", [
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
])
def test_end2end_dtypes_str_npz(
    method: str,
    tmpdir: Union[str, Path],
) -> None:
    end2end_str(
        tmpdir=tmpdir,
        method=method,
        shard_file_type="npz",
        compression="ZIP",
    )


# TODO common fixture fb_dataset
@pytest.mark.parametrize("method", [
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
])
def test_end2end_dtypes_str_fb(
    method: str,
    tmpdir: Union[str, Path],
) -> None:
    end2end_str(
        tmpdir=tmpdir,
        method=method,
        shard_file_type="fb",
        compression="LZ4",
    )
