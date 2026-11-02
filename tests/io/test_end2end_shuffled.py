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
from typing import Union

import numpy as np
import numpy.typing as npt

import sedpack
from sedpack.io import Dataset, Metadata
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT
from sedpack.io.utils import is_module_present


def end2end(tmpdir: Union[str, Path], dtype: npt.DTypeLike, method: str,
            shard_file_type: ShardFileTypeT, compression: CompressionT) -> None:
    # Super small shards (16 examples per shard, 64 total across 4 shards)
    # to ensure intra-shard shuffling and small shuffle buffers are tested.
    array_of_values = np.random.random((64, 138))
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
        examples_per_shard=16,
        shard_file_type=shard_file_type,
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
    dataset = Dataset(tiny_experiment_path)
    dataset.check()

    def iterate_sequence(shuffle: int,
                         shards: int | None = None) -> list[bytes]:
        seq: list[bytes] = []
        match method:
            case "as_tfdataset":
                for example in dataset.as_tfdataset(
                        split=TRAIN_SPLIT,
                        shards=shards,
                        shuffle=shuffle,
                        repeat=False,
                        batch_size=1,
                ):
                    seq.append(example["attribute_name"].numpy()[0].tobytes())
            case "as_numpy_iterator":
                for example in dataset.as_numpy_iterator(
                        split=TRAIN_SPLIT,
                        shards=shards,
                        shuffle=shuffle,
                        repeat=False,
                ):
                    seq.append(example["attribute_name"].tobytes())
            case "as_numpy_iterator_concurrent":
                for example in dataset.as_numpy_iterator_concurrent(
                        split=TRAIN_SPLIT,
                        shards=shards,
                        shuffle=shuffle,
                        repeat=False,
                ):
                    seq.append(example["attribute_name"].tobytes())
            case "as_numpy_iterator_rust":
                for example in dataset.as_numpy_iterator_rust(
                        split=TRAIN_SPLIT,
                        shards=shards,
                        shuffle=shuffle,
                        repeat=False,
                ):
                    seq.append(example["attribute_name"].tobytes())
            case "as_numpy_iterator_rust_batched":
                for batch in dataset.as_numpy_iterator_rust_batched(
                        split=TRAIN_SPLIT,
                        shards=shards,
                        shuffle=shuffle,
                        repeat=False,
                        batch_size=4,
                ):
                    for item in batch["attribute_name"]:
                        seq.append(item.tobytes())
            case _:
                raise ValueError(f"Unknown method: {method}")
        return seq

    expected_all = [
        array_of_values[i].tobytes() for i in range(array_of_values.shape[0])
    ]
    expected_first_shard = expected_all[:16]

    # 1. Deterministic iteration (shuffle=0) must reproduce exact order.
    assert iterate_sequence(shuffle=0) == expected_all
    assert iterate_sequence(shuffle=0, shards=1) == expected_first_shard

    # 2. Shuffled iteration across all shards with a small shuffle buffer (8):
    # two runs must yield all examples in two different orders.
    run1_all = iterate_sequence(shuffle=8)
    run2_all = iterate_sequence(shuffle=8)
    assert set(run1_all) == set(expected_all)
    assert set(run2_all) == set(expected_all)
    assert run1_all != expected_all
    assert run2_all != expected_all
    assert run1_all != run2_all

    # 3. Shuffled iteration on a single small shard (shards=1, 16 examples)
    # with a small shuffle buffer (8): even without multiple shards to shuffle,
    # intra-shard shuffling must produce two different sequences.
    run1_single = iterate_sequence(shuffle=8, shards=1)
    run2_single = iterate_sequence(shuffle=8, shards=1)
    assert set(run1_single) == set(expected_first_shard)
    assert set(run2_single) == set(expected_first_shard)
    assert run1_single != expected_first_shard
    assert run2_single != expected_first_shard
    assert run1_single != run2_single


@pytest.mark.skipif(
    not is_module_present("tensorflow"),
    reason="TensorFlow is optional, skip test if not present.",
)
def test_end2end_as_tfdataset_tfrec(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_tfdataset",
            shard_file_type="tfrec",
            compression="GZIP")


@pytest.mark.skipif(
    not is_module_present("tensorflow"),
    reason="TensorFlow is optional, skip test if not present.",
)
def test_end2end_as_tfdataset_fb(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_tfdataset",
            shard_file_type="fb",
            compression="LZ4")


@pytest.mark.skipif(
    not is_module_present("tensorflow"),
    reason="TensorFlow is optional, skip test if not present.",
)
def test_end2end_as_numpy_iterator_concurrent_tfrec(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_concurrent",
            shard_file_type="tfrec",
            compression="GZIP")


@pytest.mark.skipif(
    not is_module_present("tensorflow"),
    reason="TensorFlow is optional, skip test if not present.",
)
def test_end2end_as_numpy_iterator_tfrec(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator",
            shard_file_type="tfrec",
            compression="GZIP")


def test_end2end_as_numpy_iterator_concurrent_npz(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_concurrent",
            shard_file_type="npz",
            compression="ZIP")


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


def test_end2end_as_numpy_iterator_rust_fb(tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_rust",
            shard_file_type="fb",
            compression="LZ4")


def test_end2end_as_numpy_iterator_rust_batched_fb(
        tmpdir: Union[str, Path]) -> None:
    end2end(tmpdir=tmpdir,
            dtype="float32",
            method="as_numpy_iterator_rust_batched",
            shard_file_type="fb",
            compression="LZ4")
