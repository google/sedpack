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

import pytest

from sedpack.io.itertools import LazyPool


def test_doc() -> None:

    def f(x: int) -> int:
        return 2 * x

    with LazyPool(5) as pool:
        result = set(pool.imap_unordered(f, range(10)))

    assert result == set(f(i) for i in range(10))


def test_break_from_loop() -> None:
    f = lambda x: list(range(x))

    with LazyPool(5) as pool:
        for l in pool.imap_unordered(f, range(1_000**5)):
            if len(l) > 50:
                break
    # Not hanging.


def test_two_functions() -> None:
    with LazyPool(3) as pool:

        def f(x: int) -> tuple[int, int]:
            return (x, x)

        result = set(pool.imap_unordered(f, range(10)))
        print(result)
        assert result == set((i, i) for i in range(10))

        def g(x: int) -> str:
            return f"{x} is a number"

        result: set[str] = set(  # type: ignore[no-redef]
            pool.imap_unordered(g, range(10)))
        assert result == set(f"{i} is a number" for i in range(10)
                         )  # type: ignore[comparison-overlap]


def test_break_from_loop_break_and_iterate_again() -> None:
    f = lambda x: list(range(x))

    pool = LazyPool(5)

    with pool as pool:
        seen: int = 0
        for l in pool.imap_unordered(f, range(1_000**5)):
            seen += 1
            if len(l) >= 50:
                break
        assert seen == 51

    with pool as pool:
        seen = 0
        for l in pool.imap_unordered(f, range(1_000**5)):
            seen += 1
            if len(l) >= 50:
                break
        assert seen == 51


def test_no_restart_while_imap() -> None:
    f = lambda x: list(range(x))

    pool = LazyPool(5)

    with pool as pool:
        seen: int = 0
        for l in pool.imap_unordered(f, range(1_000**5)):
            seen += 1
            if len(l) >= 50:
                break
        assert seen == 51

        # Did not finish the neither previous iteration nor the context.
        with pytest.raises(AssertionError):
            for l in pool.imap_unordered(f, range(10)):
                pass
