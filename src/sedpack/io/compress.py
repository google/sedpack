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
"""Open a file with specified compression.
"""

import bz2
import gzip
import lz4.frame
import lzma
from pathlib import Path
from typing import IO

from sedpack.io.types import CompressionT


class CompressedFile:
    """Provide an easy open function for dealing with compressed files.
    """

    def __init__(self, compression_type: CompressionT) -> None:
        """Initialize a compressed file opening.

        compression_type (CompressionT): The type of compression. Note that ZIP
        is not supported yet.
        """
        self.compression_type: CompressionT = compression_type

        if compression_type in ["ZIP"]:
            # Zip is a container meaning we open something.zip and inside that
            # we open file(-s). This requires more work on the context manager
            # side. Not implementing yet.
            raise NotImplementedError(f"Compression {compression_type} is not "
                                      f"supported yet by CompressedFile")

    def open(self,
             file: Path | str,
             mode: str,
             encoding: str | None = None) -> IO:
        """Open function.

        Args:

          file (path-like object): The file to be opened.

          mode (str): Opening mode see `open`. The mode argument can be any of
          "r", "rb", "w", "wb", "x", "xb", "a" or "ab" for binary mode, or
          "rt", "wt", "xt", or "at" for text mode. The default is "rb".

          encoding (str | None): Encoding in case the file is not compressed
          and one uses text mode. Defaults to None.

        Example use (the expected use is opening in the binary mode):
        ```
        with CompressedFile("LZMA").open("my_file.txt", "w") as f:
          f.write("hello compressed file")
        ```
        """
        match self.compression_type:
            case "":
                # No compression.
                return open(file=file, mode=mode, encoding=encoding)
            case "GZIP" | "ZLIB":
                return gzip.open(
                    filename=file,
                    mode=mode,
                    compresslevel=9,  # slow write, but large compression
                )
            case "BZ2":
                return bz2.open(
                    filename=file,
                    mode=mode,
                    compresslevel=9,
                )
            case "LZMA":
                return lzma.open(
                    filename=file,
                    mode=mode,
                    # 0-9, default=6 more compression, but more RAM
                    # preset=None,
                )
            case "LZ4":
                return lz4.frame.open(file, mode=mode)
            case _:
                raise NotImplementedError(f"CompressedFile does not implement "
                                          f"{self.compression_type} yet.")

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return ["", "BZ2", "GZIP", "LZMA", "LZ4", "ZLIB"]

    def compress(self, data: bytes) -> bytes:
        """Self-standing compression. This is useful for instance when writing
        files using async IO.

        Args:

          data (bytes): Content to compress.

        Returns: the compressed data.
        """
        match self.compression_type:
            case "":
                return data
            case "GZIP" | "ZLIB":
                return gzip.compress(data)
            case "BZ2":
                return bz2.compress(data)
            case "LZMA":
                return lzma.compress(data)
            case "LZ4":
                return lz4.frame.compress(data)
            case _:
                raise NotImplementedError(f"CompressedFile does not implement "
                                          f"{self.compression_type} yet.")

    def decompress(self, data: bytes) -> bytes:
        """Self-standing decompression. This is useful for instance when
        reading files using async IO.

        Args:

          data (bytes): Content of the file to be decompressed.

        Returns: the decompressed data.
        """
        match self.compression_type:
            case "":
                return data
            case "GZIP" | "ZLIB":
                return gzip.decompress(data)
            case "BZ2":
                return bz2.decompress(data)
            case "LZMA":
                return lzma.decompress(data)
            case "LZ4":
                return lz4.frame.decompress(data)
            case _:
                raise NotImplementedError(f"CompressedFile does not implement "
                                          f"{self.compression_type} yet.")
