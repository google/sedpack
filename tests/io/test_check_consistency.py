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
from typing import get_args, Union

import pytest
import numpy as np

import sedpack
from sedpack.io import Dataset
from sedpack.io import Metadata
from sedpack.io.types import CompressionT, HashChecksumT, ShardFileTypeT, TRAIN_SPLIT


def get_dataset(tmp_path: Union[str, Path], shard_file_type: ShardFileTypeT,
                compression: CompressionT,
                hash_checksums: tuple[HashChecksumT, ...]) -> Dataset:
    dtype: str = "float32"
    array_of_values = np.random.random((1024, 138))
    array_of_values = array_of_values.astype(dtype)

    tiny_experiment_path: Path = Path(tmp_path) / "check_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype=dtype,
            shape=array_of_values[0].shape,
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression=compression,
        examples_per_shard=256,
        shard_file_type=shard_file_type,
        hash_checksum_algorithms=hash_checksums,
    )

    dataset = Dataset.create(
        path=tiny_experiment_path,
        metadata=dataset_metadata,
        dataset_structure=dataset_structure,
    )

    # Fill data in the dataset

    with dataset.filler(concurrency=np.random.randint(0, 4),) as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    # Check the data is correct
    # Reopen the dataset
    return Dataset(tiny_experiment_path)


def get_hash_checksums_tuples():
    # All possible hashsums.
    yield get_args(HashChecksumT)

    # Available checksums one by one.
    for single_hash_sum in get_args(HashChecksumT):
        yield (single_hash_sum,)

    # No checksum.
    yield ()


@pytest.mark.parametrize("hash_checksums", list(get_hash_checksums_tuples()))
def test_hash_checksums_ok(
    hash_checksums: tuple[HashChecksumT, ...],
    tmp_path: Union[str, Path],
) -> None:
    dataset = get_dataset(
        tmp_path=tmp_path,
        shard_file_type="fb",  # this is independent
        compression="LZ4",  # should be independent
        hash_checksums=hash_checksums,
    )
    dataset.check()


@pytest.mark.parametrize("hash_checksums", list(get_hash_checksums_tuples()))
def test_hash_checksums_changed_shard(
    hash_checksums: tuple[HashChecksumT, ...],
    tmp_path: Union[str, Path],
) -> None:
    dataset = get_dataset(
        tmp_path=tmp_path,
        shard_file_type="fb",  # this is independent
        compression="LZ4",  # should be independent
        hash_checksums=hash_checksums,
    )

    if hash_checksums == ():
        # Not able to check anyway => not raising.
        dataset.check()
        return

    # Append a byte to a shard file.
    shard_info = next(iter(dataset.shard_info_iterator(split=None)))
    file_path: Path = shard_info.file_infos[0].file_path
    with open(dataset.path / file_path, "r+b") as f:
        f.write(b"0")

    with pytest.raises(ValueError) as err:
        dataset.check()
    assert str(file_path) in str(err.value)


@pytest.mark.parametrize("hash_checksums", list(get_hash_checksums_tuples()))
def test_hash_checksums_changed_shards_list(
    hash_checksums: tuple[HashChecksumT, ...],
    tmp_path: Union[str, Path],
) -> None:
    dataset = get_dataset(
        tmp_path=tmp_path,
        shard_file_type="fb",  # this is independent
        compression="LZ4",  # should be independent
        hash_checksums=hash_checksums,
    )

    if hash_checksums == ():
        # Not able to check anyway => not raising.
        dataset.check()
        return

    # Append newline to a shards_list file.
    file_path: Path = next(iter(
        dataset._dataset_info.splits.values())).shard_list_info_file.file_path
    with open(dataset.path / file_path, "a") as f:
        f.write("\n")

    with pytest.raises(ValueError) as err:
        dataset.check()
    assert str(file_path) in str(err.value)


@pytest.mark.parametrize("hash_checksums", list(get_hash_checksums_tuples()))
def test_hash_checksums_changed_dataset_info(
    hash_checksums: tuple[HashChecksumT, ...],
    tmp_path: Union[str, Path],
) -> None:
    dataset = get_dataset(
        tmp_path=tmp_path,
        shard_file_type="fb",  # this is independent
        compression="LZ4",  # should be independent
        hash_checksums=hash_checksums,
    )

    if hash_checksums == ():
        # Not able to check anyway => not raising.
        dataset.check()
        return

    expected = dataset.current_metadata_checksums()

    # Append newline to a shards_list file.
    file_path: Path = dataset._get_config_path(dataset.path)
    with open(file_path, "a") as f:
        f.write("\n")

    with pytest.raises(ValueError) as err:
        dataset.check(hash_checksums_values=expected)
    assert str(file_path) in str(err.value)
