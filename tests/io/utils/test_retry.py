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

import pytest
import time
import random

from sedpack.io.utils import retry


@retry(
    sleep_min_s=0.0,
    sleep_max_s=0.1,
)
def always_raise() -> None:
    raise ValueError


def test_retry_raises() -> None:
    with pytest.raises(ValueError):
        always_raise()


@retry(
    stop_after_attempt=5,
    sleep_min_s=60,
    sleep_max_s=120,
)
def success_add(a: int, b: int) -> int:
    return a + b


def test_success() -> None:
    start = time.time()
    assert success_add(3, 4) == 7
    elapsed = time.time() - start
    assert elapsed < 0.01  # s


@retry(
    stop_after_attempt=1_000,
    sleep_min_s=0.0,
    sleep_max_s=0.0,
)
def flaky_mul(a: int, b: int) -> int:
    if random.randint(0, 1):
        return a * b
    else:
        raise ValueError


def test_flaky_mul() -> None:
    start = time.time()
    assert flaky_mul(3, 4) == 12
    elapsed = time.time() - start
    assert elapsed < 1  # s


@retry
def pure_retry_max(a: int, b: int, c: int) -> int:
    return max(a, b, c)


def test_pure_retry_max() -> None:
    start = time.time()
    assert pure_retry_max(3, 7, 4) == 7
    elapsed = time.time() - start
    assert elapsed < 0.01  # s


@retry
def pure_retry_flaky_max(a: int, b: int, c: int) -> int:
    """Flaky by design, 10% of cases this test takes somewhere between a minute
    and 5 minutes.
    """
    if not random.randint(0, 3):
        raise ValueError
    return max(a, b, c)


def test_pure_retry_flaky_max() -> None:
    assert pure_retry_flaky_max(3, 7, 4) == 7


@retry
def default_decorated_fn() -> None:
    """expected default docstring"""
    pass


@retry(stop_after_attempt=3)
def nondefault_decorated_fn() -> None:
    """expected non-default docstring"""
    pass


def default_nosugar_fn() -> None:
    """nosugar expected default docstring"""
    pass


default_nosugar_fn = retry(default_nosugar_fn)


def nondefault_nosugar_fn() -> None:
    """nosugar expected non-default docstring"""
    pass


nondefault_nosugar_fn = retry(stop_after_attempt=3)(nondefault_nosugar_fn)


def test_wraps_nicely() -> None:
    # Names match.
    assert default_decorated_fn.__name__ == "default_decorated_fn"
    assert nondefault_decorated_fn.__name__ == "nondefault_decorated_fn"
    assert default_nosugar_fn.__name__ == "default_nosugar_fn"
    assert nondefault_nosugar_fn.__name__ == "nondefault_nosugar_fn"

    # Docstring matches.
    assert default_decorated_fn.__doc__ == "expected default docstring"
    assert nondefault_decorated_fn.__doc__ == "expected non-default docstring"
    assert default_nosugar_fn.__doc__ == "nosugar expected default docstring"
    assert nondefault_nosugar_fn.__doc__ == "nosugar expected non-default docstring"
