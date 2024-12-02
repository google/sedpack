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

import pytest
import numpy as np
import numpy.typing as npt

from sedpack.io.compress import CompressedFile
from sedpack.io.types import CompressionT


def test_compress_gzip_write(tmpdir: str | Path) -> None:
    file_name = tmpdir / "compressed_file"
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    # Can read what gzip writes
    with gzip.open(file_name, "wb") as f:
        f.write(payload)

    with open(file_name, "rb") as f:
        assert CompressedFile("GZIP").decompress(f.read()) == payload


def test_compress_gzip_read(tmpdir: str | Path) -> None:
    file_name = tmpdir / "compressed_file"
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    # Write is readable by gzip
    with open(file_name, "wb") as f:
        f.write(CompressedFile("GZIP").compress(payload))

    with gzip.open(file_name, "rb") as f:
        assert f.read() == payload


def test_compress_decompress_file(tmpdir: str | Path) -> None:
    file_name = tmpdir / "compressed_file"
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    for algorithm in CompressedFile.supported_compressions():
        with open(file_name, "wb") as f:
            f.write(CompressedFile(algorithm).compress(payload))

        with open(file_name, "rb") as f:
            assert CompressedFile(algorithm).decompress(f.read()) == payload


def test_compress_decompress_in_memory() -> None:
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    for algorithm in CompressedFile.supported_compressions():
        # Decompress of compress is the same.
        assert CompressedFile(algorithm).decompress(
            CompressedFile(algorithm).compress(payload)) == payload


def test_compresses() -> None:
    # This should be compressible
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    assert len(CompressedFile("GZIP").compress(payload)) < len(payload)


@pytest.mark.parametrize("compression", CompressedFile.supported_compressions())
def test_compression_works(compression: CompressionT,
                           tmpdir: Union[str, Path]) -> None:
    file_name = tmpdir / "compressed_file"
    payload = bytes(x % 13 for x in range(10 * (1024**2)))  # 10MB

    # Write is readable by gzip
    with open(file_name, "wb") as f:
        compressor = CompressedFile(compression)
        f.write(compressor.compress(payload))

    with open(file_name, "rb") as f:
        assert CompressedFile(compression).decompress(f.read()) == payload
