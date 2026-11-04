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

import asyncio
from pathlib import Path
from typing import Any, Union

import numpy as np
import pytest

import sedpack
from sedpack.io import Dataset, Metadata
from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.shard.get_shard_writer import get_shard_writer
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT
from sedpack.io.utils import is_module_present


@pytest.mark.skipif(
    not is_module_present("tensorflow"),
    reason="TensorFlow is optional, skip test if not present.",
)
def test_attribute_bytes(tmpdir: Union[str, Path]) -> None:
    array_of_values = [
        bytes(
            np.random.randint(
                0,  # low
                256,  # high
                np.random.randint(10, 1_000, (), np.int32),  # size
                np.uint8,
            )) for _ in range(138)
    ]

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset

    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="attribute_name",
            dtype="bytes",
            shape=(),  # ignored
        ),
    ]

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="GZIP",
        examples_per_shard=256,
        shard_file_type="tfrec",
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

    for i, example in enumerate(
            dataset.as_numpy_iterator(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
            )):
        if example["attribute_name"] != array_of_values[i]:
            print(example["attribute_name"])
            print(array_of_values[i])
            raise ValueError

    # We tested everything
    assert i + 1 == len(array_of_values)


@pytest.mark.parametrize(
    "shard_file_type,compression",
    [
        ("fb", "LZ4"),
        ("npz", "ZIP"),
        ("tfrec", "GZIP"),
    ],
)
def test_bytes_representations_and_int_arrays_without_extra_zeros(
    tmp_path: Path,
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
) -> None:
    """Verify that `bytes`, `b""`, `bytearray`, and `np.ndarray` of `uint8`,
    `int32`, or `int64` with byte values in `[0, 255]` are saved and loaded as
    exact byte sequences without extra zero bytes.
    """
    if shard_file_type == "tfrec" and not is_module_present("tensorflow"):
        pytest.skip("TensorFlow is optional, skip test if not present.")

    var_inputs: list[Any] = [
        b"",
        b"abc\x00\x00",
        bytearray([0, 1, 2, 255]),
        np.array([10, 20, 30, 255], dtype=np.uint8),
        np.array([0, 127, 128, 255], dtype=np.int32),
        np.array([42, 0, 255, 1, 2], dtype=np.int64),
        [5, 6, 7, 250],
    ]
    expected_var: list[bytes] = [
        b"",
        b"abc\x00\x00",
        b"\x00\x01\x02\xff",
        b"\x0a\x14\x1e\xff",
        b"\x00\x7f\x80\xff",
        b"\x2a\x00\xff\x01\x02",
        b"\x05\x06\x07\xfa",
    ]

    fixed_inputs: list[Any] = [
        b"\x00\x01\x02\x03",
        b"ab\x00\x00",
        bytearray([10, 20, 30, 40]),
        np.array([1, 2, 3, 4], dtype=np.uint8),
        np.array([0, 100, 200, 255], dtype=np.int32),
        np.array([255, 254, 253, 0], dtype=np.int64),
        [9, 8, 7, 6],
    ]
    expected_fixed: list[bytes] = [
        b"\x00\x01\x02\x03",
        b"ab\x00\x00",
        b"\x0a\x14\x1e\x28",
        b"\x01\x02\x03\x04",
        b"\x00\x64\xc8\xff",
        b"\xff\xfe\xfd\x00",
        b"\x09\x08\x07\x06",
    ]

    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(name="var_bytes", dtype="bytes", shape=()),
            Attribute(name="fixed_bytes", dtype="bytes", shape=(4,)),
        ],
        compression=compression,
        examples_per_shard=3,
        shard_file_type=shard_file_type,
    )
    dataset = Dataset.create(
        path=tmp_path / f"ds_{shard_file_type}",
        metadata=Metadata(description="Test bytes representations"),
        dataset_structure=dataset_structure,
    )

    with dataset.filler(concurrency=2) as filler:
        for var_val, fixed_val in zip(var_inputs, fixed_inputs):
            filler.write_example(
                values={
                    "var_bytes": var_val,
                    "fixed_bytes": fixed_val,
                },
                split=TRAIN_SPLIT,
            )

    dataset = Dataset(dataset.path)
    dataset.check()

    for iterator in [
            dataset.as_numpy_iterator(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
            ),
            dataset.as_numpy_iterator_concurrent(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
            ),
    ]:
        seen = list(iterator)
        assert [ex["var_bytes"] for ex in seen] == expected_var
        assert [ex["fixed_bytes"] for ex in seen] == expected_fixed

    if shard_file_type in ("fb", "npz"):

        async def collect_async() -> list[dict[str, Any]]:
            return [
                ex async for ex in dataset.as_numpy_iterator_async(
                    split=TRAIN_SPLIT,
                    shuffle=0,
                    repeat=False,
                )
            ]

        seen_async = asyncio.run(collect_async())
        assert [ex["var_bytes"] for ex in seen_async] == expected_var
        assert [ex["fixed_bytes"] for ex in seen_async] == expected_fixed

    if shard_file_type == "fb":
        seen_rust = list(
            dataset.as_numpy_iterator_rust(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
            ))
        assert [ex["var_bytes"] for ex in seen_rust] == expected_var
        assert [ex["fixed_bytes"] for ex in seen_rust] == expected_fixed

        seen_batched_var: list[bytes] = []
        seen_batched_fixed: list[bytes] = []
        for batch in dataset.as_numpy_iterator_rust_batched(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=2,
        ):
            seen_batched_var.extend(
                batch["var_bytes"])  # type: ignore[arg-type]
            seen_batched_fixed.extend(
                batch["fixed_bytes"])  # type: ignore[arg-type]
        assert seen_batched_var == expected_var
        assert seen_batched_fixed == expected_fixed

    if is_module_present("tensorflow"):
        seen_tf_var: list[bytes] = []
        seen_tf_fixed: list[bytes] = []
        for batch in dataset.as_tfdataset(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=2,
        ):
            seen_tf_var.extend(batch["var_bytes"].numpy().tolist())
            seen_tf_fixed.extend(batch["fixed_bytes"].numpy().tolist())
        assert seen_tf_var == expected_var
        assert seen_tf_fixed == expected_fixed


@pytest.mark.parametrize("invalid_value", [
    np.array([-1, 0, 1], dtype=np.int32),
    np.array([0, 256], dtype=np.int64),
    np.array([1.0, 2.0], dtype=np.float32),
    [-5, 10],
    [10, 300],
    "not_bytes",
    123,
])
def test_invalid_bytes_attribute_raises(
    tmp_path: Path,
    invalid_value: Any,
) -> None:
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(name="b", dtype="bytes", shape=()),
        ],
        shard_file_type="fb",
        compression="LZ4",
    )
    writer = get_shard_writer(
        dataset_structure=dataset_structure,
        shard_file=tmp_path / "shard.fb",
    )
    with pytest.raises(ValueError):
        writer.write({"b": invalid_value})
