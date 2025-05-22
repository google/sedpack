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
"""Information how files are saved."""

from pathlib import Path
import uuid

from pydantic import BaseModel, field_validator


class FileInfo(BaseModel):
    """Represents file information with a hash checksum(-s).

    Attributes:

        file_path (Path): The file path.  Relative to the dataset root
        directory. Prefer using `DirectoryGenerator` to generate directory
        hierarchies.

        hash_checksums (tuple[str]): Control checksums of this shard file. In
        order given by DatasetStructure.hash_checksum_algorithms. As computed
        by `dataset_lib.io.utils.hash_checksums`.
    """
    file_path: Path
    hash_checksums: tuple[str, ...] = ()

    @field_validator("file_path")
    @classmethod
    def no_directory_traversal(cls, v: Path) -> Path:
        """Make sure there is no directory traversal.

        Args:

          cls: A validator for BaseModel needs to be a classmethod.

          v (Path): The path to be checked.

        Return: the original path `v`.

        Raises: ValueError in case `v` contains "..".
        """
        if ".." in v.parts:
            raise ValueError("A .. is present in the path which could allow "
                             "directory traversal above `dataset_root_path`.")
        return v


class DirectoryGenerator:
    """Random string generator to ensure a nice tree-like directory structure.
    This means not too many files or subdirectories in a directory -- at most
    `max_branching` (except possibly for the root which can have unlimited).

    Example use:
    ```python
    # These are small values of parameters. For production one should use
    # default values.
    generator = DirectoryGenerator(
        levels=3,
        max_branching=2,
        name_length=4,
    )
    for _ in range(10):
        print(generator.get_path())
    ```
    Possible output:
    ```
    ca27/ac58/f37c107d9a3549b68fba18e031aa1cf9
    ca27/ac58/ab29ddcb515f4cf1b3651cab31490142
    ca27/f469/ca5a8cf3aad942828dd9f8e56f2fbd42
    ca27/f469/ad236737fa9041318e13ac16a6400188
    9c40/7a10/e0151d9241a049d883bd985ab256c8d2
    9c40/7a10/3e1712e509c04630b7a4f690b83d2f92
    9c40/c1d9/75fff313cd1b479dbe3beb5eb78d8598
    9c40/c1d9/aa4a91ffd1514e2d97cbeafb7cac25e0
    2f66/3b68/c1139fb2669e456a85310ae9170dc66e
    2f66/3b68/ee6dbbf7d4b24a23a2126d5bc607da16
    ```
    which results in the following directory structure:
    ```bash
    ├── 2f66
    │   └── 3b68
    │       ├── c1139fb2669e456a85310ae9170dc66e
    │       └── ee6dbbf7d4b24a23a2126d5bc607da16
    ├── 9c40
    │   ├── 7a10
    │   │   ├── 3e1712e509c04630b7a4f690b83d2f92
    │   │   └── e0151d9241a049d883bd985ab256c8d2
    │   └── c1d9
    │       ├── 75fff313cd1b479dbe3beb5eb78d8598
    │       └── aa4a91ffd1514e2d97cbeafb7cac25e0
    └── ca27
        ├── ac58
        │   ├── ab29ddcb515f4cf1b3651cab31490142
        │   └── f37c107d9a3549b68fba18e031aa1cf9
        └── f469
            ├── ad236737fa9041318e13ac16a6400188
            └── ca5a8cf3aad942828dd9f8e56f2fbd42
    ```
    """

    def __init__(self,
                 levels: int = 4,
                 max_branching: int = 1_000,
                 name_length: int = 10) -> None:
        """Initialize a top level generator. No collision detection since in
        theory there could be other files present. The last part of the path is
        interpreted as the file-name without a file-extension.

        Args:

          levels (int): How many levels of directory hierarchy to generate.
          Defaults to 4.

          max_branching (int): How many subdirectories to generate (top level
          can generate unlimited). Defaults to 1_000.

          name_length (int): Maximum length of one path part. The last part is
          length 32 always to avoid collisions in the file-name. Make sure that
          `max_branching` is not too large when selecting small `name_length`.
          The name consists of hex digits so `16**name_length` should be much
          larger than `math.sqrt(max_branching)` otherwise a collision is
          probable. See `https://en.wikipedia.org/wiki/Birthday_problem`.
        """
        assert levels >= 1
        assert max_branching > 1
        assert name_length > 1

        self.levels: int = levels
        self.max_branching: int = max_branching
        self.name_length: int = name_length

        # The top level object is allowed to have arbitrary degree and never
        # raises.
        self.is_top: bool = True
        # How many part names has this object generated. Updated by `_get_name`.
        self.generated: int = 0
        # The current part name. Regenerated by `_get_name`.
        self.current_directory: str

        self.next_level: DirectoryGenerator | None = None
        if levels > 1:
            self._refresh()

    def get_path(self) -> str:
        """Generate a new path string.

        Raises: When more than `max_branching` name parts were generated. But
        only when `not self.is_top` -- the user level object never raises.
        """
        if self.next_level:
            try:
                result = "/".join((
                    self.current_directory,
                    self.next_level.get_path(),
                ))
            except ValueError:
                self._refresh()
                return self.get_path()
        else:
            result = self._get_name()

        if not self.is_top and self.generated > self.max_branching:
            # Too many paths already generated.
            raise ValueError

        return result

    def _refresh(self) -> None:
        """Generate a new child and get a new `current_directory`.
        """
        self.next_level = DirectoryGenerator(
            levels=self.levels - 1,
            max_branching=self.max_branching,
            name_length=self.name_length,
        )
        self.next_level.is_top = False
        self.current_directory = self._get_name()

    def _get_name(self) -> str:
        """Return a new part name and increase the `generated` counter.
        """
        self.generated += 1
        if self.next_level:
            return uuid.uuid4().hex[:self.name_length]
        return uuid.uuid4().hex
