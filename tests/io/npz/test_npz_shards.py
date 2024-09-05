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

import numpy as np

from sedpack.io.metadata import Attribute, DatasetStructure
from sedpack.io.shard.shard_writer_np import ShardWriterNP
from sedpack.io.npz import IterateShardNP


def test_attribute_bytes(tmpdir: str | Path) -> None:
    shard_file = Path(tmpdir / "test_shard.npz")
    examples_per_shard = 4
    attr_1_shape = (128,)
    attr_2_shape = (8,)
    attr_1_values = np.random.uniform(size=(examples_per_shard, *attr_1_shape))
    attr_2_values = np.random.uniform(size=(examples_per_shard, *attr_2_shape))

    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(
                name="attr_1",
                dtype="float32",
                shape=attr_1_shape,
            ),
            Attribute(
                name="attr_2",
                dtype="float32",
                shape=attr_2_shape,
            ),
        ],
        compression="ZIP",
        examples_per_shard=examples_per_shard,
        shard_file_type="npz",
    )

    shard_writer_np = ShardWriterNP(
        dataset_structure=dataset_structure,
        shard_file=shard_file,
    )

    for i in range(examples_per_shard):
        shard_writer_np.write(values={
            "attr_1": attr_1_values[i],
            "attr_2": attr_2_values[i],
        })
    shard_writer_np.close()

    for i, e in enumerate(
            IterateShardNP(dataset_structure=dataset_structure,
                           process_record=None).iterate_shard(shard_file)):
        assert np.allclose(e["attr_1"], attr_1_values[i])
        assert np.allclose(e["attr_2"], attr_2_values[i])
