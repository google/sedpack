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
import lzma

import lz4.frame  # type: ignore
import zstandard as zstd

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

    @staticmethod
    def supported_compressions() -> list[CompressionT]:
        """Return a list of supported compression types.
        """
        return [
            "",
            "BZ2",
            "GZIP",
            "LZMA",
            "LZ4",
            "ZLIB",
            "ZSTD",
        ]

    def compress(self, data: bytes) -> bytes:
        """Compression.

        Args:

          data (bytes): Content to compress.

        Returns: the compressed data.
        """
        match self.compression_type:
            case "":
                return data
            case "GZIP" | "ZLIB":
                return gzip.compress(data, compresslevel=9)
            case "BZ2":
                return bz2.compress(data, compresslevel=9)
            case "LZMA":
                return lzma.compress(data)
            case "LZ4":
                return lz4.frame.compress(data)
            case "ZSTD":
                return zstd.compress(data)
            case _:
                raise NotImplementedError(f"CompressedFile does not implement "
                                          f"{self.compression_type} yet.")

    def decompress(self, data: bytes) -> bytes:
        """Decompression.

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
            case "ZSTD":
                return zstd.decompress(data)
            case _:
                raise NotImplementedError(f"CompressedFile does not implement "
                                          f"{self.compression_type} yet.")
