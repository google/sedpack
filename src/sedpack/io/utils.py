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

import functools
import hashlib
from pathlib import Path
import random
import time
from typing import (
    Callable,
    Protocol,
    ParamSpec,
    TypeVar,
)
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


def hash_checksums_from_bytes(
    file_content: bytes,
    hashes: tuple[HashChecksumT, ...],
) -> tuple[str, ...]:
    """Compute the hex-encoded hash checksums. An alternative to
    `hash_checksums` to avoid reading the file again.

    Args:

        file_content (bytes): The whole file content.

        hashes (tuple[HashChecksumT, ...]): A tuple of hash algorithm names to
        be computed.

    Returns: hex-encoded hash checksums of the file in the order given by
    `hashes`.
    """
    # Actual hash functions, same order as hashes.
    hash_functions = tuple(
        _get_hash_function(hash_name) for hash_name in hashes)

    # Update all hashes.
    for hash_function in hash_functions:
        hash_function.update(file_content)

    # Hex-encoded results, same order as hashes.
    return tuple(hash_function.hexdigest() for hash_function in hash_functions)


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
                hash_function.update(bytes(memory_view[:i]))

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
    file_path.parent.mkdir(exist_ok=True, parents=True)

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


def func_or_identity(f: Callable[..., T] | None) -> Callable[..., T]:
    """Return the function or an identity in case the argument is None.
    """
    if f is None:
        return identity
    return f


P = ParamSpec("P")
R = TypeVar("R")


def retry(
    maybe_f: Callable[P, R] | None = None,
    *,
    stop_after_attempt: int = 50,
    sleep_min_s: float = 1.0,
    sleep_max_s: float = 4.0,
) -> Callable[P, R]:
    """Retry decorator. For this simple use-case no need to add another package
    as a dependency. When more functionality would be required consider using a
    package, e.g., https://tenacity.readthedocs.io/en/latest/.

    Args:

      maybe_f (Callable[P, R] | None): The decorated function when called
      without any parameter specification or None when parameters are
      specified.

      *: The rest is kwargs.

      stop_after_attempt (int): Maximal number of attempts (at least 1).

      sleep_min_s (float): Minimal sleep time in seconds (at least 0.0).

      sleep_max_s (float): Minimal sleep time in seconds (at least
      `sleep_min_s`).

    Example use:

    ```python
    @retry
    def foo():
        if not random.randint(0, 9):
            raise ValueError
        return 0
    ```

    or with different parameters (here just single retry call)

    ```python
    @retry(stop_after_attempt=2)
    def foo():
        if not random.randint(0, 9):
            raise ValueError
        return 0
    ```
    """
    # Fix nonsense values.
    stop_after_attempt = max(stop_after_attempt, 1)
    sleep_min_s = max(sleep_min_s, 0.0)
    sleep_max_s = max(sleep_min_s, sleep_max_s)

    # Capture retrying values and take just the function to be wrapped as a
    # parameter.
    def wrapper_with_defaults(f: Callable[P, R]) -> Callable[P, R]:
        # Do not forget name and docstring.
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Keep trying without re-raising.
            for _ in range(stop_after_attempt - 1):
                try:
                    return f(*args, **kwargs)
                except:  # pylint: disable=bare-except
                    # Catch all exception silently.
                    time.sleep(random.uniform(sleep_min_s, sleep_max_s))

            # Final try with possible re-raise.
            return f(*args, **kwargs)
        return wrapper

    # See https://peps.python.org/pep-0318/#current-syntax
    if maybe_f is None:
        # When parameters are specified:
        # @retry(stop_after_attempt=10)
        # def foo():
        #    ...
        # # sugar for:
        # foo = retry(
        #     maybe_f=None,
        #     stop_after_attempt=10,
        #     sleep_min_s=1.0,
        #     sleep_max_s=4.0,
        # )(foo)
        # This case seems hard for type-checking.
        return wrapper_with_defaults  # type: ignore[return-value]
    else:
        # Default parameters:
        # @retry
        # def foo():
        #    ...
        # # sugar for:
        # foo = retry(
        #     foo,
        #     stop_after_attempt=50,
        #     sleep_min_s=1.0,
        #     sleep_max_s=4.0,
        # )
        return wrapper_with_defaults(maybe_f)
