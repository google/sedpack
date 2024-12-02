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

Format: MAJOR.MINOR.PATCH (see https://pypi.org/project/semver/ for more
  possibilities)
"""
# The version of this package is defined by rust/Cargo.toml but mypy does not
# see that.
__version__ = "0.0.7"
