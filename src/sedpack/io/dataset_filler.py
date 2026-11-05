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
"""A context manager dealing with opening new shards, part number and group
number."""

from __future__ import annotations
from collections import OrderedDict, deque
import concurrent.futures
import copy
import errno
import json
from pathlib import Path
import shutil
from types import TracebackType
from typing import Any, Type, TYPE_CHECKING, get_args

import tenacity

from sedpack.io.errors import DiskFullError
from sedpack.io.file_info import PathGenerator, FileInfo
from sedpack.io.merge_shard_infos import merge_splits_shard_infos
from sedpack.io.shard_file_metadata import ShardInfo, ShardsList, ShardListInfo
from sedpack.io.shard import Shard
from sedpack.io.types import ExampleT, SplitT
from sedpack.io.metadata import DatasetStructure

if TYPE_CHECKING:
    # Avoid cyclical import.
    from sedpack.io.dataset_writing import DatasetWriting


def _is_retryable_exception(exception: BaseException) -> bool:
    """Retry transient OSErrors, but do not retry out-of-disk-space (ENOSPC)."""
    return isinstance(exception, OSError) and exception.errno != errno.ENOSPC


@tenacity.retry(
    wait=tenacity.wait_random(min=60, max=5 * 60),  # 1 to 5 minutes
    stop=tenacity.stop_after_attempt(10),
    retry=tenacity.retry_if_exception(_is_retryable_exception),
    reraise=True,
)
def _close_shard(shard: Shard) -> ShardInfo:
    """Helper function to close a shard. This can be pickled and thus sent to
    another process.
    """
    return shard.close()


class DatasetFiller:
    """Takes care of creating new shards with the correct part_number and
    group_number.

    Main use-cases:
    1. Standard single-coordinator writing (prefer `dataset.filler()`, or
       `DatasetFiller(dataset=dataset)` with default
       `relative_path_from_split=Path(".")`):
       When `relative_path_from_split` is empty (`Path(".")` or `Path("")`),
       exiting the `with` block writes all shard metadata files and updates all
       the way up to `<split>/shards_list.json` and the root
       `dataset_info.json`.
    ```python
    # Standalone use, prefer using sedpack.io.DatasetWriting.filler instead.
    # Context manager properly opens and closes shards.
    with DatasetFiller(
        dataset=dataset,
    ) as dataset_filler:
        # Add examples, new shards are opened automatically.
        for values in examples:
            dataset_filler.write_example(values=values, split=TRAIN_SPLIT)
    ```
    2. Built-in multi-process writing:
       Prefer `dataset.write_multiprocessing(...)`, which automatically assigns
       each worker a unique non-empty `relative_path_from_split` and merges
       all worker updates into `dataset_info.json` once the workers finish.

    Advanced use-case (custom multi-worker / distributed writing):
    When running multiple `DatasetFiller` instances across custom worker
    processes or distributed jobs (without `Dataset.write_multiprocessing`),
    pass a unique, non-empty `relative_path_from_split` (e.g.,
    `Path(f"worker_{worker_id}")`) to each worker's `DatasetFiller`. Because
    `relative_path_from_split` is non-empty, exiting the `with` block writes
    and hashes the worker's shard lists only up to
    `<split>/<relative_path_from_split>/shards_list.json`, without touching
    `<split>/shards_list.json` or the root `dataset_info.json`. Once all
    workers finish, collect `get_updated_infos()` on the coordinator and call
    `dataset.write_config(updated_infos=...)`:
    ```python
    def run_worker(
        dataset_path: Path,
        worker_id: int,
        items: list[ExampleT],
    ) -> list[ShardListInfo]:
        ds = Dataset(dataset_path)
        filler = DatasetFiller(
            dataset=ds,
            relative_path_from_split=Path(f"worker_{worker_id}"),
        )
        with filler:
            for item in items:
                filler.write_example(values=item, split=TRAIN_SPLIT)
        return filler.get_updated_infos()

    # Coordinator: merge all worker ShardListInfo updates into the dataset.
    updated_infos: list[ShardListInfo] = []
    for worker_id, chunk in enumerate(chunks):
        updated_infos.extend(run_worker(dataset.path, worker_id, chunk))
    dataset.write_config(updated_infos=updated_infos)
    ```

    Caveats for the advanced use-case:
    - `relative_path_from_split` MUST be non-empty for every concurrent worker
      even when each worker writes to a different split (e.g., one worker
      writing only `"train"` and another writing only `"test"`). Leaving
      `relative_path_from_split` empty (`Path(".")`) causes `__exit__` to call
      `dataset.write_config(...)` and overwrite the shared root
      `dataset_info.json`, racing with other workers and losing their split
      updates.
    - Concurrent workers must use disjoint `relative_path_from_split` paths
      where neither path is a parent prefix of another (e.g., do not run
      `Path("a")` and `Path("a/b")` concurrently, because the `Path("a")`
      worker writes `<split>/a/shards_list.json` on exit). Sibling paths such
      as `Path("job_1/worker_0")` and `Path("job_1/worker_1")` are safe.
    - When `relative_path_from_split` is non-empty, written shards are not
      registered in `<split>/shards_list.json` or `dataset_info.json` until the
      coordinator calls `dataset.write_config(updated_infos=...)` with the
      `ShardListInfo` objects returned by `filler.get_updated_infos()`.
    """

    def __init__(
        self,
        dataset: DatasetWriting,
        concurrency: int = 4,
        relative_path_from_split: Path = Path("."),
        *,
        max_disk_utilization: float | None = 0.8,
    ) -> None:
        """Context manager for writing examples into a dataset.

        Args:

          dataset (DatasetWriting): Dataset object representing the dataset we
          are filling.

          concurrency (int): Setting to a positive integer allows writing shard
          files in parallel using a `ProcessPoolExecutor`. Defaults to 4.

          relative_path_from_split (Path): New shards are created inside
          `dataset_root_path / split / relative_path_from_split` or children.
          It will be created if it does not exist yet. When empty (`Path(".")`,
          the default), `__exit__` updates metadata all the way up to
          `<split>/shards_list.json` and `dataset_info.json`. When non-empty
          (e.g., `Path("worker_0")`), `__exit__` updates metadata only up to
          `<split>/<relative_path_from_split>/shards_list.json` so multiple
          workers can run without race conditions; the caller must then pass
          `get_updated_infos()` to `dataset.write_config(updated_infos=...)`
          (even if each worker writes to a different split). May not contain
          "..".

          max_disk_utilization (float | None): Maximum fraction of disk capacity
          in `(0.0, 1.0]` allowed before raising `DiskFullError`, or `None` to
          disable the disk utilization check. Defaults to `0.8` (80%). Checked
          at creation (`__init__`) and in `write_example` before opening a new
          shard. When `DiskFullError` is raised during `write_example`,
          `__exit__` still flushes and saves all so-far opened shards and
          updates the dataset metadata so no previously written data is lost
          (which may cause final disk utilization to slightly exceed
          `max_disk_utilization`).

        Raises:

          ValueError: If `relative_path_from_split` contains '..' or
          `max_disk_utilization` is outside `(0.0, 1.0]`.

          DiskFullError: If disk utilization is already `>=
          max_disk_utilization` at creation time.
        """
        # Make sure that we do not accidentally traverse the directory above
        # `dataset_root_path`.
        self._relative_path_from_split: Path = FileInfo.no_directory_traversal(
            relative_path_from_split)

        if (max_disk_utilization is not None and
                not 0.0 < max_disk_utilization <= 1.0):
            raise ValueError(
                "max_disk_utilization must be in the range (0.0, 1.0] or None, "
                f"got {max_disk_utilization}")

        self._dataset: DatasetWriting = dataset
        self._dataset_root_path: Path = dataset.path
        self._dataset_structure: DatasetStructure = dataset.dataset_structure
        self._max_disk_utilization: float | None = max_disk_utilization

        # Check disk space before spawning the executor pool so we do not leak
        # background processes if creation fails.
        self._check_disk_space()

        self._max_workers: int = max(1, concurrency)
        self._concurrent_pool: concurrent.futures.Executor | None = None

        # Constants.
        self._examples_per_shard: int = (
            self._dataset_structure.examples_per_shard)

        # LRU cache of currently open shards keyed by `(split, metadata_key)`.
        # Bounded to at most `self._max_workers` open shards simultaneously.
        self._current_shards: OrderedDict[tuple[SplitT, str],
                                          Shard] = OrderedDict()

        # Cumulated shard lists keyed by relative path to `shards_list.json`.
        self._shards_lists: dict[Path, ShardsList] = {}
        self._future_shard_infos: deque[
            concurrent.futures.Future[ShardInfo]] = deque()
        self._updated_infos_by_split: dict[SplitT, ShardListInfo] = {}

        # Random path generator.
        self._path_generator = PathGenerator()

    def _check_disk_space(self) -> None:
        """Check that disk utilization is below `self._max_disk_utilization`.

        Disk utilization is computed as
        `(usage.total - usage.free) / usage.total` so that filesystem blocks
        reserved for `root` (e.g., 5% on Linux ext4) are counted as unavailable
        to non-root processes (`1.0` corresponds to `usage.free == 0`).

        Raises:

          DiskFullError: If current disk utilization is `>=
          self._max_disk_utilization`.
        """
        if self._max_disk_utilization is None:
            return

        check_path: Path = self._dataset_root_path
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent

        usage = shutil.disk_usage(check_path)
        if usage.total <= 0:
            return
        current_utilization: float = (usage.total - usage.free) / usage.total
        if current_utilization >= self._max_disk_utilization:
            raise DiskFullError(
                dataset_path=self._dataset_root_path,
                message=(
                    f"Disk utilization {current_utilization:.1%} at "
                    f"'{self._dataset_root_path}' reached or exceeded "
                    f"max_disk_utilization={self._max_disk_utilization:.1%}."),
                current_utilization=current_utilization,
                max_disk_utilization=self._max_disk_utilization,
            )

    def _ensure_pool(self) -> concurrent.futures.Executor:
        if self._concurrent_pool is None:
            self._concurrent_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._max_workers)
        return self._concurrent_pool

    def _record_shard_info(self, shard_info: ShardInfo) -> None:
        """Record a completed `ShardInfo` into `self._shards_lists`."""
        if shard_info.number_of_examples <= 0:
            return
        relative_path_self: Path = (shard_info.file_infos[0].file_path.parent /
                                    "shards_list.json")
        if relative_path_self not in self._shards_lists:
            self._shards_lists[relative_path_self] = ShardsList.load_or_create(
                dataset_root_path=self._dataset_root_path,
                relative_path_self=relative_path_self,
            )
        shards_list = self._shards_lists[relative_path_self]
        shards_list.shard_files.append(shard_info)
        shards_list.number_of_examples += shard_info.number_of_examples

    @property
    def shard_lists(self) -> dict[Path, ShardsList]:
        """Return information about pending leaf shard lists buffered in this
        `DatasetFiller` before `_update_infos()` flushes them on `__exit__`.
        """
        while self._future_shard_infos:
            self._record_shard_info(self._future_shard_infos.popleft().result())
        return self._shards_lists

    def _get_new_shard(self, split: SplitT) -> Shard:
        # Relative path to the shard file.
        relative_path_with_split: Path = split / self._relative_path_from_split
        # Create new shard info.
        file_type: str = self._dataset_structure.shard_file_type

        deeper_path: Path = self._path_generator.get_path().with_suffix(
            "." + file_type)
        shard_info = ShardInfo(file_infos=(FileInfo(
            file_path=relative_path_with_split / deeper_path),))

        return Shard(
            shard_info=shard_info,
            dataset_structure=self._dataset_structure,
            dataset_root_path=self._dataset_root_path,
        )

    def write_example(
        self,
        values: ExampleT,
        split: SplitT,
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write an example. Opens a new shard if necessary.

        Up to `concurrency` (`max(1, concurrency)`) shards are kept open
        concurrently across `(split, custom_metadata)` pairs using an LRU
        policy, so oscillating `custom_metadata` or `split` values reuse open
        shards instead of creating single-example shards. Examples without
        `custom_metadata` (`None` or `{}`) and examples with non-empty
        `custom_metadata` are always saved in separate shards.

        Args:

            values (ExampleT): A dictionary with all values.

            split (SplitT): Which split to write this example into.

            custom_metadata (dict[str, Any] | None): Optional JSON-serializable
            metadata saved in the shard's `ShardInfo`. Shards can later be
            filtered using these metadata.

        Raises:

            DiskFullError: If a new shard needs to be opened and disk
            utilization is at or above `max_disk_utilization`. When used inside
            a `DatasetFiller` context manager, all shards opened so far are
            still closed and saved to the dataset on context exit (`__exit__`)
            before the exception propagates.
        """
        if split not in get_args(SplitT):
            raise ValueError(f"Not a known split value: {split}")

        metadata_key: str = (json.dumps(custom_metadata, sort_keys=True)
                             if custom_metadata else "{}")
        shard_key: tuple[SplitT, str] = (split, metadata_key)

        current_shard: Shard | None = self._current_shards.get(shard_key)
        if current_shard is not None:
            current_shard.write(values=values)
            if (current_shard.shard_info.number_of_examples
                    >= self._examples_per_shard):
                self.close_shard(shard=self._current_shards.pop(shard_key))
            else:
                self._current_shards.move_to_end(shard_key)
            return

        if len(self._current_shards) >= self._max_workers:
            _, evicted_shard = self._current_shards.popitem(last=False)
            self.close_shard(shard=evicted_shard)

        self._check_disk_space()

        current_shard = self._get_new_shard(split=split)
        if custom_metadata:
            current_shard.shard_info.custom_metadata = copy.deepcopy(
                custom_metadata)

        current_shard.write(values=values)

        if (current_shard.shard_info.number_of_examples
                >= self._examples_per_shard):
            self.close_shard(shard=current_shard)
        else:
            self._current_shards[shard_key] = current_shard

    def close_shard(
        self,
        shard: Shard,
        split: SplitT | None = None,
    ) -> None:
        """Close shard asynchronously in the process pool. Called automatically
        when a shard is full, evicted, or in `DatasetFiller.__exit__`.
        """
        _ = split
        if shard.shard_info.number_of_examples <= 0:
            return

        while (len(self._future_shard_infos) >= self._max_workers or
               (self._future_shard_infos and
                self._future_shard_infos[0].done())):
            self._record_shard_info(self._future_shard_infos.popleft().result())

        self._future_shard_infos.append(self._ensure_pool().submit(
            _close_shard,
            shard=shard,
        ))

    def __enter__(self) -> DatasetFiller:
        """Initialize the process pool for shard writing."""
        self._ensure_pool()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Make sure to close all open shards and update metadata files.

        Args:

          exc_type (Type[BaseException] | None): None if no exception,
          otherwise the exception type.

          exc_value (BaseException | None): None if no exception, otherwise
          the exception value.

          exc_tb (TracebackType | None): None if no exception, otherwise the
          traceback.
        """
        try:
            # Close each open shard that has at least one example written.
            while self._current_shards:
                _, shard = self._current_shards.popitem(last=False)
                if shard.shard_info.number_of_examples > 0:
                    self.close_shard(shard=shard)
        finally:
            # Wait to finish writing of all files (after closing all shards but
            # before updating metadata files), then release the pool so
            # DatasetFiller is pickleable again.
            if self._concurrent_pool is not None:
                self._concurrent_pool.shutdown(
                    wait=True,
                    cancel_futures=False,
                )
                self._concurrent_pool = None

        # Write updated info files.
        self._update_infos()

        if not self._relative_path_from_split.parts:
            # When relative_path_from_split is empty, update all the way up to
            # `<split>/shards_list.json` and `dataset_info.json`.
            self._dataset.write_config(updated_infos=self.get_updated_infos())

    def _update_infos(self) -> None:
        """Write leaf `shards_list.json` files for all newly closed shards and
        merge them up to `<split>/<relative_path_from_split>/shards_list.json`.
        """
        while self._future_shard_infos:
            self._record_shard_info(self._future_shard_infos.popleft().result())

        if not self._shards_lists:
            return

        # Write leaf shards_list.json files and clear the pending buffer so
        # subsequent context manager runs only flush newly written shards.
        leaf_infos: list[ShardListInfo] = [
            shards_list.write_config(
                dataset_root_path=self._dataset_root_path,
                hashes=self._dataset_structure.hash_checksum_algorithms,
            ) for shards_list in self._shards_lists.values()
        ]
        self._shards_lists.clear()

        # Merge leaf shard lists up to
        # `<split>/<relative_path_from_split>/shards_list.json`.
        self._updated_infos_by_split.update(
            merge_splits_shard_infos(
                updates=leaf_infos,
                dataset_root=self._dataset_root_path,
                hashes=self._dataset_structure.hash_checksum_algorithms,
                relative_path_from_split=self._relative_path_from_split,
            ))

    def get_updated_infos(self) -> list[ShardListInfo]:
        """Return list of `ShardListInfo` objects to update the dataset (one
        per written split pointing to
        `<split>/<relative_path_from_split>/shards_list.json`).
        """
        return list(self._updated_infos_by_split.values())
