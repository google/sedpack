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
from typing import Union

import numpy as np
import numpy.typing as npt
import pytest

import sedpack
from sedpack.io import Dataset, DatasetFiller
from sedpack.io import Metadata
from sedpack.io.file_info import PathGenerator
from sedpack.io.shard_info_iterator import (
    CachedShardInfoIterator,
    ShardInfoIterator,
)
from sedpack.io.types import (
    CompressionT,
    HOLDOUT_SPLIT,
    ShardFileTypeT,
    SplitT,
    TEST_SPLIT,
    TRAIN_SPLIT,
)
from sedpack.io.utils import is_module_present


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


@pytest.mark.parametrize("single_process", [True, False])
def test_write_multiprocessing(
    tmpdir: Union[str, Path],
    single_process: bool,
) -> None:
    dtype = "float32"
    array_of_values = np.random.random((1024, 138))
    array_of_values = array_of_values.astype(dtype)

    tiny_experiment_path: Path = (Path(tmpdir) /
                                  f"write_multiprocessing_{single_process}")

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
        shard_file_type="fb",
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

    results = dataset.write_multiprocessing(
        feed_writer=feed_writer,
        custom_arguments=custom_arguments,
        custom_kwarguments=custom_kwarguments,
        single_process=single_process,
    )

    assert results == [
        len(part_of_array_of_values[0])
        for part_of_array_of_values in custom_arguments
    ]

    # Check the data is correct

    for i, example in enumerate(
            dataset.as_numpy_iterator_rust_batched(
                split=TRAIN_SPLIT,
                shuffle=0,
                repeat=False,
                batch_size=1,
            )):
        assert np.allclose(example["attribute_name"], array_of_values[i:i + 1])

    # We tested everything
    assert i + 1 == array_of_values.shape[
        0], "Not all examples have been iterated"


@pytest.mark.parametrize(
    "shard_file_type,compression",
    [
        ("fb", "LZ4"),
        ("npz", "ZIP"),
        ("tfrec", "GZIP"),
    ],
)
def test_filler_multiprocess_all_shard_types(
    tmpdir: Union[str, Path],
    shard_file_type: ShardFileTypeT,
    compression: CompressionT,
) -> None:
    if shard_file_type == "tfrec" and not is_module_present("tensorflow"):
        pytest.skip("TensorFlow is optional, skip test if not present.")

    array_of_values = np.random.random((60, 16)).astype(np.float32)
    dataset_path = Path(tmpdir) / f"filler_mp_{shard_file_type}"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test multiprocess filler"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="attribute_name",
                    dtype="float32",
                    shape=(16,),
                ),
            ],
            compression=compression,
            examples_per_shard=10,
            shard_file_type=shard_file_type,
        ),
    )

    with dataset.filler(concurrency=3) as filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)

    iterated = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(iterated) == len(array_of_values)
    np.testing.assert_allclose(np.stack(iterated), array_of_values)


def test_filler_multiple_leaf_directories_and_validation(
    tmpdir: Union[str, Path],) -> None:
    array_of_values = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    dataset_path = Path(tmpdir) / "filler_multi_leaf"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test multiple leaf directories"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="attribute_name",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    with pytest.raises(ValueError, match="directory traversal"):
        DatasetFiller(
            dataset=dataset,
            relative_path_from_split=Path("../escape"),
        )
    with pytest.raises(ValueError, match="directory traversal"):
        DatasetFiller(
            dataset=dataset,
            relative_path_from_split=Path("/tmp/absolute_escape"),
        )

    filler = dataset.filler(concurrency=2)
    # Force PathGenerator to roll over to a new directory every 2 shards.
    filler._path_generator = PathGenerator(max_branching=2, levels=2)

    with filler:
        for attribute_value in array_of_values:
            filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )
        # Exercise the shard_lists property before exit (5 full shards across
        # 3 leaf directories: 2, 2, 1 shards per leaf directory).
        assert len(filler.shard_lists) == 3

    # After __exit__, _update_infos merges the 3 leaf directories into a single
    # ShardListInfo for `train/shards_list.json`.
    updated_infos = filler.get_updated_infos()
    assert len(updated_infos) == 1
    assert updated_infos[0].shard_list_info_file.file_path == Path(
        "train/shards_list.json")
    assert updated_infos[0].number_of_examples == 20
    assert updated_infos[0].number_of_shards == 5

    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)
    iterated = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(iterated) == len(array_of_values)
    np.testing.assert_allclose(np.stack(iterated), array_of_values)


def test_filler_relative_path_from_split_updates_only_subtree(
    tmpdir: Union[str, Path],) -> None:
    """When `relative_path_from_split` is non-empty, exiting `DatasetFiller`
    updates up to `<split>/<relative_path_from_split>/shards_list.json` without
    modifying `<split>/shards_list.json` or `dataset_info.json` until
    `dataset.write_config(updated_infos=...)` is called.
    """
    array_of_values = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    dataset_path = Path(tmpdir) / "filler_subtree_only"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test subtree-only update"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="attribute_name",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    worker_0_filler = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=2,
        relative_path_from_split=Path("worker_0"),
    )
    # Force multiple leaf directories under train/worker_0/.
    worker_0_filler._path_generator = PathGenerator(max_branching=2, levels=2)

    with worker_0_filler:
        for attribute_value in array_of_values[:12]:
            worker_0_filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    worker_0_infos = worker_0_filler.get_updated_infos()
    assert len(worker_0_infos) == 1
    assert worker_0_infos[0].shard_list_info_file.file_path == Path(
        "train/worker_0/shards_list.json")
    assert worker_0_infos[0].number_of_examples == 12
    assert (dataset_path / "train" / "worker_0" / "shards_list.json").is_file()
    # Neither `<split>/shards_list.json` nor `dataset_info.json` splits should
    # be updated yet.
    assert not (dataset_path / "train" / "shards_list.json").exists()
    assert not Dataset(dataset_path)._dataset_info.splits

    worker_1_filler = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=2,
        relative_path_from_split=Path("worker_1"),
    )
    with worker_1_filler:
        for attribute_value in array_of_values[12:]:
            worker_1_filler.write_example(
                values={"attribute_name": attribute_value},
                split=TRAIN_SPLIT,
            )

    worker_1_infos = worker_1_filler.get_updated_infos()
    assert len(worker_1_infos) == 1
    assert worker_1_infos[0].shard_list_info_file.file_path == Path(
        "train/worker_1/shards_list.json")
    assert worker_1_infos[0].number_of_examples == 8

    # Coordinator merges worker updates into `<split>/shards_list.json` and
    # `dataset_info.json`.
    dataset.write_config(updated_infos=worker_0_infos + worker_1_infos)
    assert (dataset_path / "train" / "shards_list.json").is_file()

    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)
    iterated = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(iterated) == len(array_of_values)
    np.testing.assert_allclose(np.stack(iterated), array_of_values)


def test_filler_relative_path_multi_split_nested_and_append(
    tmpdir: Union[str, Path],) -> None:
    """Test nested `relative_path_from_split` across multiple splits and
    subsequent appends to an existing worker directory.
    """
    train_values = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
    test_values = (np.arange(8 * 4, dtype=np.float32).reshape(8, 4) + 100.0)
    val_values = (np.arange(6 * 4, dtype=np.float32).reshape(6, 4) + 200.0)
    dataset_path = Path(tmpdir) / "filler_nested_multi_split"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test nested multi-split workers"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="attribute_name",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    # Worker 0 writes to both TRAIN_SPLIT and TEST_SPLIT under job_1/worker_0.
    w0 = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=1,
        relative_path_from_split=Path("job_1/worker_0"),
    )
    with w0:
        for val in train_values[:6]:
            w0.write_example(values={"attribute_name": val}, split=TRAIN_SPLIT)
        for val in test_values:
            w0.write_example(values={"attribute_name": val}, split=TEST_SPLIT)

    w0_infos = w0.get_updated_infos()
    assert {info.shard_list_info_file.file_path for info in w0_infos} == {
        Path("train/job_1/worker_0/shards_list.json"),
        Path("test/job_1/worker_0/shards_list.json"),
    }

    # Worker 1 writes to TRAIN_SPLIT and HOLDOUT_SPLIT under job_1/worker_1.
    w1 = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=1,
        relative_path_from_split=Path("job_1/worker_1"),
    )
    with w1:
        for val in train_values[6:12]:
            w1.write_example(values={"attribute_name": val}, split=TRAIN_SPLIT)
        for val in val_values:
            w1.write_example(values={"attribute_name": val},
                             split=HOLDOUT_SPLIT)

    w1_infos = w1.get_updated_infos()
    assert {info.shard_list_info_file.file_path for info in w1_infos} == {
        Path("train/job_1/worker_1/shards_list.json"),
        Path("holdout/job_1/worker_1/shards_list.json"),
    }

    dataset.write_config(updated_infos=w0_infos + w1_infos)

    # Now append more examples into the existing job_1/worker_0 directory.
    w0_append = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=1,
        relative_path_from_split=Path("job_1/worker_0"),
    )
    with w0_append:
        for val in train_values[12:]:
            w0_append.write_example(
                values={"attribute_name": val},
                split=TRAIN_SPLIT,
            )

    dataset.write_config(updated_infos=w0_append.get_updated_infos())

    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)

    iterated_train = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    iterated_test = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TEST_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    iterated_val = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=HOLDOUT_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]

    expected_train = np.concatenate(
        [train_values[:6], train_values[12:], train_values[6:12]], axis=0)
    np.testing.assert_allclose(np.stack(iterated_train), expected_train)
    np.testing.assert_allclose(np.stack(iterated_test), test_values)
    np.testing.assert_allclose(np.stack(iterated_val), val_values)


def test_filler_reentrant_empty_and_sequential_updates(
    tmpdir: Union[str, Path],) -> None:
    """Test empty `with filler:` exits, re-entering the same `DatasetFiller`
    instance multiple times, passing multiple sequential updates for the same
    worker directory to `write_config`, and iterating an unwritten split.
    """
    values = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    dataset_path = Path(tmpdir) / "filler_reentrant_and_empty"

    dataset = Dataset.create(
        path=dataset_path,
        metadata=Metadata(description="Test reentrant and empty filler"),
        dataset_structure=sedpack.io.metadata.DatasetStructure(
            saved_data_description=[
                sedpack.io.metadata.Attribute(
                    name="attribute_name",
                    dtype="float32",
                    shape=(4,),
                ),
            ],
            compression="LZ4",
            examples_per_shard=4,
            shard_file_type="fb",
        ),
    )

    # Iterating an unwritten split (even with repeat=True) yields 0 shards
    # instead of raising KeyError or IndexError.
    assert list(
        ShardInfoIterator(
            dataset_path=dataset.path,
            dataset_info=dataset._dataset_info,
            split=HOLDOUT_SPLIT,
            repeat=False,
        )) == []
    assert list(
        CachedShardInfoIterator(
            dataset_path=dataset.path,
            dataset_info=dataset._dataset_info,
            split=HOLDOUT_SPLIT,
            repeat=True,
        )) == []

    # Exiting a filler without writing any examples is a clean no-op.
    empty_filler = DatasetFiller(
        dataset=dataset,
        relative_path_from_split=Path("empty_worker"),
    )
    with empty_filler:
        pass
    assert empty_filler.get_updated_infos() == []
    assert not (dataset_path / "train" / "empty_worker").exists()

    # Re-entering the same DatasetFiller instance across two `with` blocks and
    # passing both `get_updated_infos()` snapshots into a single `write_config`
    # call works without assertion errors or duplicate counting.
    worker_filler = DatasetFiller(
        dataset=Dataset(dataset_path),
        concurrency=1,
        relative_path_from_split=Path("job_1/worker_0"),
    )
    with worker_filler:
        for val in values[:6]:
            worker_filler.write_example(
                values={"attribute_name": val},
                split=TRAIN_SPLIT,
            )
    first_infos = worker_filler.get_updated_infos()

    with worker_filler:
        for val in values[6:]:
            worker_filler.write_example(
                values={"attribute_name": val},
                split=TRAIN_SPLIT,
            )
    second_infos = worker_filler.get_updated_infos()

    dataset.write_config(updated_infos=first_infos + second_infos)
    reloaded = Dataset(dataset_path)
    reloaded.check(show_progressbar=False)
    iterated = [
        ex["attribute_name"] for ex in reloaded.as_numpy_iterator(
            split=TRAIN_SPLIT,
            shuffle=0,
            repeat=False,
        )
    ]
    assert len(iterated) == len(values)
    np.testing.assert_allclose(np.stack(iterated), values)
