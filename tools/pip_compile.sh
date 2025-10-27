#!/usr/bin/env bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Make sure uv is installed.
python -m pip install uv

export MIN_PYTHON_VERSION=$(python tools/get_min_required_version.py)

# PLATFORM as used by GitHub workflows RUNNER_OS:
# https://docs.github.com/en/actions/reference/workflows-and-actions/variables#default-environment-variables
for PLATFORM in Linux Windows macOS
do
	# Compile dev dependencies.
	# PYTHON_PLATFORM as used by uv pip compile https://docs.astral.sh/uv/reference/cli/#uv-pip-compile
	export PYTHON_PLATFORM=$(echo ${PLATFORM} | tr '[:upper:]' '[:lower:]')
	python -m uv pip compile \
		--python-platform ${PYTHON_PLATFORM} \
		--python-version ${MIN_PYTHON_VERSION} \
		--generate-hashes \
		--upgrade \
		--all-extras \
		pyproject.toml > requirements/dev_${PLATFORM}_requirements.txt

	# Compile minimal dependencies.
	# PYTHON_PLATFORM as used by uv pip compile https://docs.astral.sh/uv/reference/cli/#uv-pip-compile
	export PYTHON_PLATFORM=$(echo ${PLATFORM} | tr '[:upper:]' '[:lower:]')
	python -m uv pip compile \
		--python-platform ${PYTHON_PLATFORM} \
		--python-version ${MIN_PYTHON_VERSION} \
		--generate-hashes \
		--upgrade \
		pyproject.toml > requirements/${PLATFORM}_requirements.txt
done
