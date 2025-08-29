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
from typing import Any
import random

import numpy as np
import numpy.typing as npt
import pytest

import sedpack
from sedpack.io import Dataset
from sedpack.io.shard_info_iterator import ShardInfoIterator
from sedpack.io import Metadata
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT


def dataset_and_values_dynamic_shape(
    tmpdir: str | Path,
    shard_file_type: str,
    compression: str,
    dtypes: list[str],
    items: int,
) -> (Dataset, dict[str, list[Any]]):
    values: dict[str, list[Any]] = {}
    ds_path = Path(tmpdir) / f"e2e_{shard_file_type}_{'_'.join(dtypes)}"
    dataset_metadata = Metadata(description="Test of the lib")

    # The order should not play a role.
    random.shuffle(dtypes)

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name=f"attribute_{dtype}",
            dtype=dtype,
            shape=(),
        ) for dtype in dtypes
    ]

    for dtype in dtypes:
        values[f"attribute_{dtype}"] = []

        match dtype:
            case "int":
                for _ in range(items):
                    # TODO larger range than just int64
                    values[f"attribute_{dtype}"].append(
                        random.randint(-2**60, 2**60))
            case "str":
                long_string = "Ḽơᶉëᶆ ȋṕšᶙṁ ḍỡḽǭᵳ ʂǐť ӓṁệẗ, ĉṓɲṩḙċťᶒțûɾ" \
                      "https://arxiv.org/abs/2306.07249 ḹẩḇőꝛế" \
                      "ấɖḯƥĭṩčįɳġ ḝłįʈ, șếᶑ ᶁⱺ ẽḭŭŝḿꝋď ṫĕᶆᶈṓɍ ỉñḉīḑȋᵭṵńť ṷŧ" \
                      ":(){ :|:& };: éȶ đꝍꞎôꝛȇ ᵯáꞡᶇā ąⱡîɋṹẵ."
                for _ in range(items):
                    begin: int = random.randint(0, len(long_string) // 2)
                    end: int = random.randint(begin + 1, len(long_string))
                    values[f"attribute_{dtype}"].append(long_string[begin:end])
            case "bytes":
                for _ in range(items):
                    values[f"attribute_{dtype}"].append(
                        np.random.randint(
                            0,
                            256,
                            size=random.randint(5, 20),
                            dtype=np.uint8,
                        ).tobytes())

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
        path=ds_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset

    with dataset.filler() as filler:
        for i in range(items):
            filler.write_example(
                values={
                    name: value[i] for name, value in values.items()
                },
                split=TRAIN_SPLIT,
            )

    # Check the data is correct
    # Reopen the dataset
    dataset = Dataset(ds_path)
    dataset.check()

    return (values, dataset)


@pytest.fixture(
    scope="module",
    params=[
        {
            "dtypes": ["str"],
            "compression": "GZIP",
        },
        {
            "dtypes": ["bytes"],
            "compression": "GZIP",
        },
        {
            "dtypes": ["int"],
            "compression": "GZIP",
        },
        {
            "dtypes": ["str", "bytes", "int"],
            "compression": "GZIP",
        },
    ],
)
def values_and_dataset_tfrec(request, tmpdir_factory) -> None:
    shard_file_type: str = "tfrec"
    yield dataset_and_values_dynamic_shape(
        tmpdir=tmpdir_factory.mktemp(f"dtype_{shard_file_type}"),
        shard_file_type=shard_file_type,
        compression=request.param["compression"],
        dtypes=request.param["dtypes"],
        items=137,
    )
    # Teardown.


@pytest.fixture(
    scope="module",
    params=[
        {
            "dtypes": ["str"],
            "compression": "ZIP",
        },
        {
            "dtypes": ["bytes"],
            "compression": "ZIP",
        },
        {
            "dtypes": ["int"],
            "compression": "ZIP",
        },
        {
            "dtypes": ["str", "bytes", "int"],
            "compression": "ZIP",
        },
    ],
)
def values_and_dataset_npz(request, tmpdir_factory) -> None:
    shard_file_type: str = "npz"
    yield dataset_and_values_dynamic_shape(
        tmpdir=tmpdir_factory.mktemp(f"dtype_{shard_file_type}"),
        shard_file_type=shard_file_type,
        compression=request.param["compression"],
        dtypes=request.param["dtypes"],
        items=137,
    )
    # Teardown.


@pytest.fixture(
    scope="module",
    params=[
        {
            "dtypes": ["str"],
            "compression": "LZ4",
        },
        {
            "dtypes": ["bytes"],
            "compression": "LZ4",
        },
        {
            "dtypes": ["int"],
            "compression": "LZ4",
        },
        {
            "dtypes": ["str", "bytes", "int"],
            "compression": "LZ4",
        },
    ],
)
def values_and_dataset_fb(request, tmpdir_factory) -> None:
    shard_file_type: str = "fb"
    yield dataset_and_values_dynamic_shape(
        tmpdir=tmpdir_factory.mktemp(f"dtype_{shard_file_type}"),
        shard_file_type=shard_file_type,
        compression=request.param["compression"],
        dtypes=request.param["dtypes"],
        items=137,
    )
    # Teardown.


def check_iteration_of_values(
    method: str,
    dataset: Dataset,
    values: dict[str, list[Any]],
) -> None:
    match method:
        case "as_tfdataset":
            for i, example in enumerate(
                    dataset.as_tfdataset(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                        batch_size=1,
                    )):
                assert len(example) == len(values)

                # No idea how to have an actual string or bytes in TensorFlow.
                # Maybe it is best to leave it as a tensor anyway since that is
                # the "native" type.

                for name, returned_batch in example.items():
                    assert returned_batch == values[name][i:i + 1]
        case "as_numpy_iterator":
            for i, example in enumerate(
                    dataset.as_numpy_iterator(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert len(example) == len(values)
                for name, returned_value in example.items():
                    if dataset.dataset_structure.shard_file_type != "tfrec":
                        assert returned_value == values[name][i]
                        assert type(returned_value) == type(values[name][i])
                    else:
                        if "attribute_str" == name:
                            assert returned_value == values[name][i].encode(
                                "utf-8")
                        else:
                            assert returned_value == values[name][i]
        case "as_numpy_iterator_concurrent":
            for i, example in enumerate(
                    dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert len(example) == len(values)
                for name, returned_value in example.items():
                    if dataset.dataset_structure.shard_file_type != "tfrec":
                        assert returned_value == values[name][i]
                        assert type(returned_value) == type(values[name][i])
                    else:
                        if "attribute_str" == name:
                            assert returned_value == values[name][i].encode(
                                "utf-8")
                        else:
                            assert returned_value == values[name][i]
        case "as_numpy_iterator_rust":
            for i, example in enumerate(
                    dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shuffle=0,
                        repeat=False,
                    )):
                assert len(example) == len(values)
                for name, returned_value in example.items():
                    assert returned_value == values[name][i]
                    assert type(returned_value) == type(values[name][i])

    # We tested everything
    if i + 1 != len(next(iter(values.values()))):
        raise AssertionError("Not all examples have been iterated")

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


@pytest.mark.parametrize("method", [
    "as_tfdataset",
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
])
def test_end2end_dtypes_str_tfrec(
    method: str,
    values_and_dataset_tfrec,
) -> None:
    values, dataset = values_and_dataset_tfrec
    check_iteration_of_values(
        method=method,
        dataset=dataset,
        values=values,
    )


@pytest.mark.parametrize("method", [
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
])
def test_end2end_dtypes_str_npz(
    method: str,
    values_and_dataset_npz,
) -> None:
    values, dataset = values_and_dataset_npz
    check_iteration_of_values(
        method=method,
        dataset=dataset,
        values=values,
    )


@pytest.mark.parametrize("method", [
    "as_numpy_iterator",
    "as_numpy_iterator_concurrent",
    "as_numpy_iterator_rust",
])
def test_end2end_dtypes_str_fb(
    method: str,
    values_and_dataset_fb,
) -> None:
    values, dataset = values_and_dataset_fb
    check_iteration_of_values(
        method=method,
        dataset=dataset,
        values=values,
    )
