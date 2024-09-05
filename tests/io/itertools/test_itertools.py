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
import random
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from sedpack.io.itertools import *


def test_round_robin_simple() -> None:
    l = ["ABC", "D", "EFGH"]

    assert sorted(round_robin(l)) == list("ABCDEFGH")


def test_round_robin_long_iter() -> None:
    l = map(lambda x: range(x, x + 10), range(100))

    assert len(list(round_robin(l))) == 1_000


def test_round_robin_docstring() -> None:
    l = ["ABC", "D", "EF"]

    assert sorted(round_robin(l)) == ["A", "B", "C", "D", "E", "F"]


def test_random_shuffle() -> None:
    random.seed(42)
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    seen = []

    for x in shuffle_buffer(l, buffer_size=3):
        seen.append(x)

    # We have seen all elements and each only once.
    assert sorted(l) == sorted(seen)
    assert l != seen
