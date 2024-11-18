# Copyright 2023-2024 Google LLC
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
"""Utils for sedpack.io"""

import hashlib
from pathlib import Path
import time
from typing import Callable, Protocol, TypeVar
import uuid

import xxhash

from sedpack.io.file_info import FileInfo
from sedpack.io.types import HashChecksumT


class HashProtocol(Protocol):
    """Represent a hash function protocol.
    """

    def update(self, buffer: bytes) -> None:
        ...

    def digest(self) -> bytes:
        ...

    def hexdigest(self) -> str:
        ...


def _get_hash_function(name: HashChecksumT) -> HashProtocol:
    """Get a hash function by name.

    Args:

      name (HashChecksumT): Name of the hash function (string literal).

    Returns: An instance of HashProtocol.
    """
    match name:
        case "xxh32":
            return xxhash.xxh32()
        case "xxh64":
            return xxhash.xxh64()
        case "xxh128":
            return xxhash.xxh128()
        case _:
            # No other hashes are supported yet.
            return hashlib.new(name)


def hash_checksums(file_path: Path, hashes: tuple[HashChecksumT,
                                                  ...]) -> tuple[str, ...]:
    """Compute the hex-encoded hash checksums.

    Args:

        file_path (Path): Path to the file.

        hashes (tuple[HashChecksumT, ...]): A tuple of hash algorithm names to
        be computed.

    Returns: hex-encoded hash checksums of the file in the order given by
    `hashes`.
    """
    # Actual hash functions, same order as hashes.
    hash_functions = tuple(
        _get_hash_function(hash_name) for hash_name in hashes)

    memory_view = memoryview(bytearray(128 * 1024))
    with open(file_path, "rb", buffering=0) as hashed_file:
        # Read as long as we read something.
        for i in iter(lambda: hashed_file.readinto(memory_view), 0):
            # Update all hashes.
            for hash_function in hash_functions:
                hash_function.update(memory_view[:i])

    # Hex-encoded results, same order as hashes.
    return tuple(hash_function.hexdigest() for hash_function in hash_functions)


def safe_update_file(dataset_root_path: Path, relative_path: Path, info: str,
                     hashes: tuple[HashChecksumT, ...]) -> FileInfo:
    """Safely update a file and fill it with `info` in the JSON format.

    Args:

      dataset_root_path (Path): Path where the dataset is saved.

      relative_path (Path): Relative path from the dataset root.

      info (str): JSON serialized string.

      hashes (tuple[HashChecksumT, ...]): Which hashes to compute (can be empty
      if we do not need the output).

    TODO: This should be regarded as a workaround until a better versioning
    of metadata is found.

    Returns: A FileInfo object representing this file.
    """
    file_path = dataset_root_path / relative_path

    # Create parent directory if it does not exist yet.
    file_path.parent.mkdir(exist_ok=True)

    # Create the new file (unique, but not a tmp file so that it stays within
    # the same directory).
    update_id = f"{time.time()}_{uuid.uuid4().hex}"
    new_file = file_path.parent / f"update_{update_id}_of_{file_path.name}"

    # Write info into it.
    with open(new_file, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(info)

    # Replace the old file with the new version.
    new_file.replace(file_path)

    # Return the new file info.
    return FileInfo(
        file_path=relative_path,
        hash_checksums=hash_checksums(file_path=file_path, hashes=hashes),
    )


T = TypeVar("T")


def identity(x: T) -> T:
    """Identity function, because pylint was complaining about
    unnecessary-lambda-assignment.
    """
    return x


def func_or_identity(f: Callable | None) -> Callable:
    """Return the function or an identity in case the argument is None.
    """
    if f is None:
        return identity
    return f
