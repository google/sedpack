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

import gzip
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from sedpack.io.utils import hash_checksums


def test_compress_gzip_write(tmpdir: str | Path) -> None:
    file_name = tmpdir / "file.txt"
    hashes = ("md5", "sha256", "sha512", "sha384")
    with open(file_name, "w", encoding="ascii") as f:
        f.write("Hello world")

    checksums = hash_checksums(file_name, hashes)

    assert checksums == (
        "3e25960a79dbc69b674cd4ec67a72c62",
        "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c",
        "b7f783baed8297f0db917462184ff4f08e69c2d5e5f79a942600f9725f58ce1f29c18139bf80b06c0fff2bdd34738452ecf40c488c22a7e3d80cdf6f9c1c0d47",
        "9203b0c4439fd1e6ae5878866337b7c532acd6d9260150c80318e8ab8c27ce330189f8df94fb890df1d298ff360627e1",
    )
