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

import itertools

from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.shard_info_iterator.balanced_iterator import _split_balancing


def test_split_balancing_all() -> None:
    shard_list: list[ShardInfo] = [
        ShardInfo(
            file_infos=(),
            number_of_examples=1,
            custom_metadata={
                "id": i,
                "id_mod_3_is_zero": i % 3 == 0,
            },
        ) for i in range(100)
    ]

    balanced = _split_balancing(
        shard_list=shard_list,
        balance_by=[
            lambda shard_info: shard_info.custom_metadata["id_mod_3_is_zero"]
        ],
        repeat=False,
        shuffle=10,
    )

    assert set(
        shard_info.custom_metadata["id"] for shard_info in balanced) == set(
            shard_info.custom_metadata["id"] for shard_info in shard_list)


def test_split_balancing_balances() -> None:
    shard_list: list[ShardInfo] = [
        ShardInfo(
            file_infos=(),
            number_of_examples=1,
            custom_metadata={
                "id": i,
                "id_mod_3_is_zero": i % 3 == 0,
            },
        ) for i in range(100)
    ]

    balanced = _split_balancing(
        shard_list=shard_list,
        balance_by=[
            lambda shard_info: shard_info.custom_metadata["id_mod_3_is_zero"]
        ],
        repeat=True,
        shuffle=10,
    )

    take_n = 1_000
    assert sum(shard_info.custom_metadata["id_mod_3_is_zero"] for shard_info in
               list(itertools.islice(balanced, take_n))) == take_n // 2


def test_custom_weight() -> None:
    shard_list: list[ShardInfo] = [
        ShardInfo(
            file_infos=(),
            number_of_examples=1,
            custom_metadata={
                "id": i,
                "id_mod_3_is_zero": i % 3 == 0,
            },
        ) for i in range(100)
    ]

    class BalanceBy:

        def __call__(self, shard_info: ShardInfo) -> bool:
            return shard_info.custom_metadata["id_mod_3_is_zero"]

        def weight(self, shard_info: ShardInfo) -> float:
            if shard_info.custom_metadata["id_mod_3_is_zero"]:
                # Do four times more of the zeros. Meaning for each non-zero
                # example there are four zero examples -> 80% of the zero
                # examples.
                return 4
            else:
                return 1

    balance_by_top = BalanceBy()

    balanced = _split_balancing(
        shard_list=shard_list,
        balance_by=[
            balance_by_top,
        ],
        repeat=True,
        shuffle=10,
    )

    take_n = 1_000
    assert sum(shard_info.custom_metadata["id_mod_3_is_zero"] for shard_info in
               list(itertools.islice(balanced, take_n))) == take_n * (4 / 5)
