# Copyright 2023 Google LLC
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
"""Dataset library.

Version format: MAJOR.MINOR.PATCH (see https://pypi.org/project/semver/ for
more possibilities).
"""

from importlib.metadata import version, PackageNotFoundError

# The version of this package is defined by rust/Cargo.toml and dynamically
# deduced by Maturin (see
# https://www.maturin.rs/metadata.html#dynamic-metadata). To ensure
# compatibility with existing code we dynamically set the __version__ attribute
# here.
try:
    # When package is installed use the version.
    __version__ = version("sedpack")
except PackageNotFoundError:
    # Package is not installed. The Rust part of this package is probably not
    # going to work in this case (the Rust binding would be probably missing).
    __version__ = "0.0.7-dev"
