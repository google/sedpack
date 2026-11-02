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
import pytest
from typing import Union

import numpy as np

import sedpack
from sedpack.io import Dataset, Metadata
from sedpack.io.iteration import RustGenerator
from sedpack.io.metadata import DatasetStructure
from sedpack.io.types import TRAIN_SPLIT


def test_wrong_file_paralelism() -> None:
    with pytest.raises(
            ValueError,
            match="The argument file_parallelism should be positive.*",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(),
            shard_iterator=[],
            process_record=None,
            file_parallelism=0,
        )


def test_wrong_shard_type() -> None:
    with pytest.raises(
            ValueError,
            match="RustGenerator is implemented only for FlatBuffers.",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(shard_file_type="tfrec"),
            shard_iterator=[],
            process_record=None,
            file_parallelism=1,
        )


def test_wrong_compression() -> None:
    with pytest.raises(
            ValueError,
            match=
            "The compression .* is not among the supported compressions: .*",
    ):
        g = RustGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(
                shard_file_type="fb",
                compression="ZIP",
            ),
            shard_iterator=[],
            process_record=None,
            file_parallelism=1,
        )


def test_as_numpy_iterator_rust_shuffled_and_deterministic(
        tmpdir: Union[str, Path]) -> None:
    array_of_values = np.arange(64, dtype=np.int64).reshape((64, 1))
    dataset_path = Path(tmpdir) / "rust_gen_test"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test RustGenerator shuffle"),
        dataset_structure=DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="val",
                    dtype="int64",
                    shape=(1,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=16,
            shard_file_type="fb",
        ),
    )
    with dataset.filler() as filler:
        for v in array_of_values:
            filler.write_example(values={"val": v}, split=TRAIN_SPLIT)

    dataset = Dataset(dataset_path)

    # Deterministic iteration (shuffle=0) across all shards.
    det1 = [
        int(ex["val"][0]) for ex in dataset.as_numpy_iterator_rust(
            split=TRAIN_SPLIT,
            repeat=False,
            shuffle=0,
        )
    ]
    det2 = [
        int(ex["val"][0]) for ex in dataset.as_numpy_iterator_rust(
            split=TRAIN_SPLIT,
            repeat=False,
            shuffle=0,
        )
    ]
    assert det1 == list(range(64))
    assert det2 == list(range(64))

    # Shuffled iteration on a single small shard (16 examples) with small
    # shuffle buffer (shuffle=8): two runs must produce two different sequences
    # and neither may equal the deterministic sequence.
    shuf1 = [
        int(ex["val"][0]) for ex in dataset.as_numpy_iterator_rust(
            split=TRAIN_SPLIT,
            shards=1,
            repeat=False,
            shuffle=8,
        )
    ]
    shuf2 = [
        int(ex["val"][0]) for ex in dataset.as_numpy_iterator_rust(
            split=TRAIN_SPLIT,
            shards=1,
            repeat=False,
            shuffle=8,
        )
    ]
    assert sorted(shuf1) == list(range(16))
    assert sorted(shuf2) == list(range(16))
    assert shuf1 != list(range(16))
    assert shuf2 != list(range(16))
    assert shuf1 != shuf2
