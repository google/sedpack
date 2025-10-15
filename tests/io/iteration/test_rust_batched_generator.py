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
import random
from typing import Union
import uuid

import numpy as np

import sedpack
from sedpack.io import Dataset
from sedpack.io.iteration import RustBatchedGenerator
from sedpack.io.metadata import DatasetStructure, Metadata
from sedpack.io.shard_info_iterator import CachedShardInfoIterator
from sedpack.io.types import TRAIN_SPLIT, CompressionT, ShardFileTypeT


@pytest.fixture(scope="module")
def dataset_and_values(tmpdir_factory) -> None:
    data_points: int = 1_024
    dtype: str = "float32"

    # Values saved in the dataset.
    values = {
        "fixed":
            np.random.random((data_points, 138)).astype(dtype),
        "fixed_2d":
            np.random.random((data_points, 3, 5)).astype(dtype),
        "dynamic_shape_bytes": [
            uuid.uuid4().hex[:random.randint(11, 19)].encode("ascii")
            for _ in range(data_points)
        ],
        "dynamic_shape_int": [
            random.randint(-2**60, 2**60) for _ in range(data_points)
        ],
        "dynamic_shape_str": [
            uuid.uuid4().hex[:random.randint(15, 25)]
            for _ in range(data_points)
        ],
    }
    tmpdir = tmpdir_factory.mktemp("end_2_end_data")

    tiny_experiment_path: Path = Path(tmpdir) / "e2e_experiment"

    # Create a dataset
    dataset_metadata = Metadata(description="Test of the lib")

    example_attributes = [
        sedpack.io.metadata.Attribute(
            name="fixed",
            dtype=str(dtype),
            shape=values["fixed"][0].shape,
        ),
        sedpack.io.metadata.Attribute(
            name="fixed_2d",
            dtype=str(dtype),
            shape=values["fixed_2d"][0].shape,
        ),
        sedpack.io.metadata.Attribute(
            name="dynamic_shape_bytes",
            dtype="bytes",
            shape=(),
        ),
        sedpack.io.metadata.Attribute(
            name="dynamic_shape_str",
            dtype="str",
            shape=(),
        ),
        sedpack.io.metadata.Attribute(
            name="dynamic_shape_int",
            dtype="int",
            shape=(),
        ),
    ]
    random.shuffle(example_attributes)

    dataset_structure = sedpack.io.metadata.DatasetStructure(
        saved_data_description=example_attributes,
        compression="LZ4",
        examples_per_shard=24,
        shard_file_type="fb",
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

    with dataset.filler(concurrency=np.random.randint(0, 4),) as filler:
        for i in range(data_points):
            filler.write_example(
                values={
                    name: val[i] for name, val in values.items()
                },
                split=TRAIN_SPLIT,
            )

    # Check the data is correct
    # Reopen the dataset
    dataset = Dataset(tiny_experiment_path)
    dataset.check()

    yield (dataset, values)

    # Teardown


def test_wrong_file_paralelism() -> None:
    with pytest.raises(
            ValueError,
            match="The argument file_parallelism should be positive.*",
    ):
        g = RustBatchedGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(),
            shard_iterator=[],
            process_batch=None,
            file_parallelism=0,
            batch_size=1,
        )


def test_wrong_shard_type() -> None:
    with pytest.raises(
            ValueError,
            match="RustBatchedGenerator is implemented only for FlatBuffers.",
    ):
        g = RustBatchedGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(shard_file_type="tfrec"),
            shard_iterator=[],
            process_batch=None,
            file_parallelism=1,
            batch_size=1,
        )


def test_wrong_compression() -> None:
    with pytest.raises(
            ValueError,
            match=
            "The compression .* is not among the supported compressions: .*",
    ):
        g = RustBatchedGenerator(
            dataset_path=Path(),
            dataset_structure=DatasetStructure(
                shard_file_type="fb",
                compression="ZIP",
            ),
            shard_iterator=[],
            process_batch=None,
            file_parallelism=1,
            batch_size=1,
        )


# We are testing that the batches are not original order -- need at least
# batch_size > 10 for this to be not flaky.
@pytest.mark.parametrize("batch_size", [10])
def test_end_to_end_rust_batched_shuffled(
    batch_size,
    dataset_and_values,
):
    dataset, values = dataset_and_values

    remembered_values = {name: [] for name in values}

    with RustBatchedGenerator(
            dataset_path=dataset.path,
            dataset_structure=dataset.dataset_structure,
            shard_iterator=CachedShardInfoIterator(
                dataset_path=dataset.path,
                dataset_info=dataset.dataset_info,
                split="train",
                repeat=False,
                shards=None,
                custom_metadata_type_limit=None,
                shard_filter=None,
                shuffle=0,  # Shard shuffle could introduce unwanted randomness.
            ),
            batch_size=batch_size,
            process_batch=None,
            file_parallelism=8,
            shuffle_buffer_size=10 * batch_size,
    ) as g:
        index: int = 0
        for batch in g():
            current_batch_size: int = -1

            # Check that the batch is not deterministic.
            for name, attribute_values in batch.items():
                if current_batch_size < 0:
                    current_batch_size = len(attribute_values)
                else:
                    assert len(attribute_values) == current_batch_size

                for i in range(current_batch_size):
                    if name.startswith("dynamic_shape"):
                        if values[name][index + i] != attribute_values[i]:
                            break
                    else:
                        if (values[name][index + i]
                                != attribute_values[i]).all():
                            break
                else:
                    raise ValueError("This batch was deterministic")

            index += current_batch_size

            # Remember seen values to compare later.
            for name, attribute_values in batch.items():
                remembered_values[name].extend(attribute_values)

    # Test that we have seen everything.
    for name, original in values.items():
        original = np.sort(original, axis=0)
        batched = np.sort(remembered_values[name], axis=0)
        np.testing.assert_equal(batched, original)


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_end_to_end_rust_batched(
    batch_size,
    dataset_and_values,
):
    dataset, values = dataset_and_values

    with RustBatchedGenerator(
            dataset_path=dataset.path,
            dataset_structure=dataset.dataset_structure,
            shard_iterator=CachedShardInfoIterator(
                dataset_path=dataset.path,
                dataset_info=dataset.dataset_info,
                split="train",
                repeat=False,
                shards=None,
                custom_metadata_type_limit=None,
                shard_filter=None,
                shuffle=0,
            ),
            batch_size=batch_size,
            process_batch=None,
            file_parallelism=8,
    ) as g:
        index: int = 0
        for batch in g():
            current_batch_size: int = -1

            for name, attribute_values in batch.items():
                if current_batch_size < 0:
                    current_batch_size = len(attribute_values)
                else:
                    assert len(attribute_values) == current_batch_size

                for i in range(current_batch_size):
                    if name.startswith("dynamic_shape"):
                        assert values[name][index + i] == attribute_values[i]
                    else:
                        assert (values[name][index +
                                             i] == attribute_values[i]).all()

            index += current_batch_size


@pytest.mark.parametrize("batch_size", [1, 3])
def test_end_to_end_as_numpy_iterator_rust(
    batch_size,
    dataset_and_values,
):
    dataset, values = dataset_and_values
    index: int = 0

    for batch in dataset.as_numpy_iterator_rust_batched(
            split="train",
            process_batch=None,
            shards=None,
            shard_filter=None,
            repeat=False,
            batch_size=batch_size,
            file_parallelism=8,
            shuffle=0,
    ):
        current_batch_size: int = -1

        for name, attribute_values in batch.items():
            if current_batch_size < 0:
                current_batch_size = len(attribute_values)
            else:
                assert len(attribute_values) == current_batch_size

            for i in range(current_batch_size):
                if name.startswith("dynamic_shape"):
                    assert values[name][index + i] == attribute_values[i]
                else:
                    assert (values[name][index +
                                         i] == attribute_values[i]).all()

        index += current_batch_size
