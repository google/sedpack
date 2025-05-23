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

from sedpack.io.file_info import PathGenerator


def test_multilevel() -> None:
    levels: int = 3
    max_branching: int = 2
    name_length: int = 3

    generator = PathGenerator(
        levels=levels,
        max_branching=max_branching,
        name_length=name_length,
    )

    seen_paths: set[str] = set()

    # No exception in the top level
    for _ in range((max_branching**levels) + 10):
        p = generator.get_path()
        assert len(p.parts) == levels

        # Last part is long, previous are at most name_length
        assert len(p.parts[-1]) >= 32
        assert all(len(part) == name_length for part in p.parts[:-1])

        seen_paths.add(p)

    # Enforce bounded number of subdirectories except the top
    for l in range(levels - 1):
        for p in seen_paths:
            prefix = p.parents[l]
            count = sum(path.is_relative_to(prefix) for path in seen_paths)
            assert count <= max_branching**(l + 1)


def test_single_level() -> None:
    max_branching: int = 3
    name_length: int = 3
    generator = PathGenerator(
        levels=1,
        max_branching=max_branching,
        name_length=name_length,
    )

    # No exception in the top level
    for _ in range(max_branching + 10):
        n = generator.get_path()
        assert len(str(n)) >= 32  # last level is unbounded length
        assert len(n.parts) == 1
