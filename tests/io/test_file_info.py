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

from sedpack.io.file_info import DirectoryGenerator


def test_multilevel() -> None:
    levels: int = 3
    max_branching: int = 2
    name_length: int = 3

    generator = DirectoryGenerator(
        levels=levels,
        max_branching=max_branching,
        name_length=name_length,
    )

    seen_names: set[str] = set()

    # No exception in the top level
    for _ in range((max_branching**levels) + 10):
        n = generator.get_path()
        assert n.count("/") == levels - 1
        p = Path(n)
        assert len(p.parts) == levels

        # Last part is long, previous are at most name_length
        assert len(p.parts[-1]) >= 32
        assert all(len(part) == name_length for part in p.parts[:-1])

        # Enforce format: name/name/name/long_name
        for l in range(1, levels):
            assert n[l * (1 + name_length) - 1] == "/"

        seen_names.add(n)

    # Enforce tree structure
    for l in range(1, levels):
        for n in seen_names:
            prefix = n[:l * (1 + name_length) - 1]
            count = 0
            for name in seen_names:
                if name.startswith(prefix):
                    count += 1
            assert count <= max_branching**(levels - l)


def test_single_level() -> None:
    max_branching: int = 3
    name_length: int = 3
    generator = DirectoryGenerator(
        levels=1,
        max_branching=max_branching,
        name_length=name_length,
    )

    # No exception in the top level
    for _ in range(max_branching + 10):
        n = generator.get_path()
        assert len(n) >= 32  # last level is unbounded length
        assert "/" not in n
