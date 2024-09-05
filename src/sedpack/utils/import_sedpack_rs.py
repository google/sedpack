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

def import_sedpack_rs():
    """Try to import the Rust library. If that fails return None. TODO: this
    should return an object which would raise an error when accessing the
    result.
    """
    try:
        import sedpack_rs
        return sedpack_rs
    except ImportError:
        return None
