# Copyright 2023-2025 Google LLC
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
"""Custom exceptions for sedpack.io."""

import errno
from pathlib import Path


class DatasetExistsError(FileExistsError):
    """Error for signalling that the dataset already exists."""

    def __init__(self, dataset_path: Path) -> None:
        """Represents that the dataset already exists.

        Args:
          dataset_path: The dataset path.
        """
        super().__init__(
            f'Dataset info file exists and would be overwritten. Use instead:'
            f' Dataset(path="{dataset_path}")')
        self.dataset_path = dataset_path


class DiskFullError(OSError):
    """Error for signalling that disk utilization is at or above the safety
    threshold.
    """

    def __init__(
        self,
        dataset_path: Path,
        message: str,
        *,
        current_utilization: float | None = None,
        max_disk_utilization: float | None = None,
    ) -> None:
        """Represents that the disk does not have enough free space.

        Args:
          dataset_path: The dataset path.
          message: Human-readable error message.
          current_utilization: Current disk utilization ratio in [0.0, 1.0].
          max_disk_utilization: Maximum allowed disk utilization ratio.
        """
        super().__init__(errno.ENOSPC, message)
        self.dataset_path = dataset_path
        self.current_utilization = current_utilization
        self.max_disk_utilization = max_disk_utilization
