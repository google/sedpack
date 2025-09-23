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
import concurrent.futures
import dataclasses
from pathlib import Path
from types import TracebackType
from typing import Any, Type, TYPE_CHECKING

from sedpack.io.file_info import PathGenerator, FileInfo
from sedpack.io.metadata import DatasetStructure
from sedpack.io.shard import Shard
from sedpack.io.shard_file_metadata import ShardInfo, ShardsList, ShardListInfo
from sedpack.io.types import ExampleT, SplitT
from sedpack.io.utils import retry

if TYPE_CHECKING:
    # Avoid cyclical import.
    from sedpack.io.dataset_writing import DatasetWriting


@retry(
    stop_after_attempt=10,
    sleep_min_s=60.0,
    sleep_max_s=180.0,
)
def _close_shard(shard: Shard) -> ShardInfo:
    """Helper function to close a shard. This can be pickled and thus sent to
    another process.
    """
    return shard.close()


@dataclasses.dataclass
class ShardProgress:
    """Internal information about a shard progress. Since we do not want to
    close a shard each time an example is written into another split we keep
    multiple shards open. This makes keeping information easier.
    """
    shard: Shard
    written_examples: int = 0


class DatasetFillerContext:
    """The actual context. Use DatasetFiller which closes the last shard on
    exit. Takes care of creating new shards with proper part_number and
    group_number.
    """

    def __init__(
        self,
        dataset_root_path: Path,
        dataset_structure: DatasetStructure,
        relative_path_from_split: Path,
        *,
        concurrent_pool: concurrent.futures.Executor,
        write_updates: bool = True,
    ) -> None:
        """Initialize a dataset filler context which writes examples and
        automatically opens and closes shards (except possibly not closing the
        last shard which is closed by `DatasetFiller`).

        Args:

          dataset_root_path (Path): Dataset root directory.

          dataset_structure (DatasetStructure): What information is saved about
          each example and how they are saved.

          relative_path_from_split (Path): New shards are created inside
          `dataset_root_path / split / relative_path_from_split`. May not
          contain "..".

          concurrent_pool (concurrent.futures.Executor): Shard file writing.

          write_updates (bool): Whether to save progress of written shard info
          files. Defaults to `True`. In theory when set to `False` writing
          could be faster (not writing the info file after each shard). But
          that way progress may be lost (e.g., when the process is interrupted
          during writing a shard file).
        """
        # Make sure that we do not accidentally traverse the directory above
        # `dataset_root_path`.
        if ".." in relative_path_from_split.parts:

            raise ValueError("The `relative_path_from_split` may not contain "
                             "'..' which could allow accidental directory "
                             "traversal.")

        self._dataset_root_path: Path = dataset_root_path
        self._dataset_structure: DatasetStructure = dataset_structure
        self._relative_path_from_split: Path = relative_path_from_split
        self._concurrent_pool: concurrent.futures.Executor = concurrent_pool
        self._write_updates: bool = write_updates

        # Constants.
        self._examples_per_shard: int = dataset_structure.examples_per_shard

        # Current shard.
        self._current_shards_progress: dict[SplitT, ShardProgress] = {}

        # Cumulated shard infos.
        self._shards_lists: dict[SplitT, ShardsList] = {}
        self._future_shard_infos: dict[
            SplitT, list[concurrent.futures.Future[ShardInfo]]] = {}

        # Random path generator.
        self._path_generator = PathGenerator()

    @property
    def shard_lists(self) -> dict[SplitT, ShardsList]:
        """Return information about all shards written by this
        DatasetFillerContext.
        """
        if not self._future_shard_infos:
            return self._shards_lists

        for split, shard_info_futures in self._future_shard_infos.items():
            for future_info in shard_info_futures:
                shard_info: ShardInfo = future_info.result()

                # Remember this shard info.
                if split not in self._shards_lists:
                    # Load if exists.
                    self._shards_lists[split] = ShardsList.load_or_create(
                        dataset_root_path=self._dataset_root_path,
                        relative_path_self=shard_info.file_infos[0].file_path.
                        parent / "shards_list.json",
                    )
                self._shards_lists[split].shard_files.append(shard_info)
                self._shards_lists[
                    split].number_of_examples += shard_info.number_of_examples

        # Write down which shard files have been saved.
        if self._write_updates:
            for shards_list in self._shards_lists.values():
                shards_list.write_config(
                    dataset_root_path=self._dataset_root_path,
                    hashes=(),  # We forget these now.
                )

        self._future_shard_infos = {}
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

    def write_example(self,
                      values: ExampleT,
                      split: SplitT,
                      custom_metadata: dict[str, Any] | None = None) -> None:
        """Write an example. Opens a new shard if necessary.

        Args:

            values (ExampleT): A dictionary with all values.

            split (SplitT): Which split to write this example into.

            custom_metadata (dict[str, Any] | None): Optional metadata saved
            with in the shard info. The shards then can be filtered using these
            metadata. When a value is changed a new shard is open and the
            current shard is closed. TODO there is no check if a shard with the
            same `custom_metadata` already exists. When a `custom_metadata` is
            provided for the very first time it is set for the current shard
            (retroactively also affecting the previous writes). This behavior
            is not ideal but a proper solution will require a proper design
            (e.g., to avoid having too many open shards, finding out how to
            find the shard with correct `custom_metadata` since a dictionary
            cannot be easily hashed, etc.).
        """
        # Check that we have a shard for the split.
        if split not in self._current_shards_progress:
            self._current_shards_progress[split] = ShardProgress(
                self._get_new_shard(split=split))

        current_progress: ShardProgress = self._current_shards_progress[split]

        # New metadata
        previous_metadata = current_progress.shard.shard_info.custom_metadata
        metadata_changed: bool = all((
            custom_metadata,
            previous_metadata,
            custom_metadata != previous_metadata,
        ))

        # Open a new shard if the current one already contains too many
        # examples.
        if (current_progress.written_examples >= self._examples_per_shard or
                metadata_changed):
            # Close the current shard if needed.
            self.close_shard(shard=current_progress.shard, split=split)
            current_progress.shard = self._get_new_shard(split=split)
            current_progress.written_examples = 0

        # Update custom_metadata is needed
        if custom_metadata:
            current_progress.shard.shard_info.custom_metadata = custom_metadata

        # Write the current example and update counters.
        current_progress.shard.write(values=values)
        current_progress.written_examples += 1

        # We have updated the current progress.
        assert self._current_shards_progress[split] == current_progress

    def close_shard(self, shard: Shard, split: SplitT) -> None:
        """Close shard. Called automatically by DatasetFiller.__exit__."""
        # Finish writing the shard.
        if split not in self._future_shard_infos:
            self._future_shard_infos[split] = []
        self._future_shard_infos[split].append(
            self._concurrent_pool.submit(
                _close_shard,
                shard=shard,
            ))


class DatasetFiller:
    """Takes care of creating new shards with the correct part_number and
    group_number.

    Example use:
    # Standalone use, prefer using sedpack.io.DatasetWriting.filler instead.
    # Context manager properly opens and closes shards.
    with DatasetFiller(
        dataset=dataset,
    ) as dataset_filler:
        # Add examples, new shards are opened automatically.
        for values in examples:
            dataset_filler.write_example(values=values)
    """

    def __init__(self,
                 dataset: DatasetWriting,
                 concurrency: int = 1,
                 relative_path_from_split: Path = Path("."),
                 auto_update_dataset: bool = True) -> None:
        """Context manager for writing examples into a dataset.

        Args:

          dataset (DatasetWriting): Dataset object representing the dataset we
          are filling.

          concurrency (int): Setting to a positive integer allows writing shard
          files in parallel. Defaults to 1 (sequential writes in another thread
          or process). Thread is used for "tfrec" otherwise process.

          relative_path_from_split (Path): New shards are created inside
          `dataset_root_path / split / relative_path_from_split` or children.
          It will be created if it does not exist yet. Useful for
          multiprocessing where we want to save changes without overwriting
          files and race conditions.

          auto_update_dataset (bool): Defaults to `True`. When closed
          (`__exit__` is called) write information about all written shards
          back to the dataset. Defaults to `True`. When one creates multiple
          `DatasetFiller` objects it is advised to set this parameter to
          `False` since this is not thread safe.
        """
        self._concurrent_pool: concurrent.futures.Executor
        if dataset.dataset_structure.shard_file_type == "tfrec":
            self._concurrent_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1)
        else:
            self._concurrent_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, concurrency),)

        self._dataset_filler_context: DatasetFillerContext
        self._dataset_filler_context = DatasetFillerContext(
            dataset_root_path=dataset.path,
            dataset_structure=dataset.dataset_structure,
            relative_path_from_split=relative_path_from_split,
            concurrent_pool=self._concurrent_pool,
        )
        self._auto_update_dataset: bool = auto_update_dataset
        self._dataset: DatasetWriting = dataset
        self._updated_infos: list[ShardListInfo] = []

    def __enter__(self) -> DatasetFillerContext:
        """Initialize DatasetFillerContext."""
        assert not self._updated_infos
        return self._dataset_filler_context

    def __exit__(self, exc_type: Type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        """Make sure to close the last shard.

        Args:

          exc_type (Type[BaseException] | None): None if no exception,
          otherwise the exception type.

          exc_value (BaseException | None): None if no exception, otherwise
          the exception value.

          exc_tb (TracebackType | None): None if no exception, otherwise the
          traceback.
        """
        # Close the shard only if there was an example written.
        split_progress = self._dataset_filler_context._current_shards_progress
        for split, shard_progress in split_progress.items():
            if shard_progress.written_examples > 0:
                # Need to do it this way when there would be a shard with less
                # than examples_per_shard examples.
                self._dataset_filler_context.close_shard(
                    shard=shard_progress.shard,
                    split=split,
                )

        # Wait to finish writing of all files (after closing all shards but
        # before updating metadata files).
        self._concurrent_pool.shutdown(
            wait=True,
            cancel_futures=False,
        )

        # Write updated info files.
        self._update_infos()

        if self._auto_update_dataset:
            # Note that when a DatasetFiller is pickled and unpickled it no
            # longer has the same `self._dataset` object.
            self._dataset.write_config(updated_infos=self._updated_infos)

    def _update_infos(self) -> None:
        """Write back all information about written shards to the dataset.
        """
        assert not self._updated_infos
        # Update shards lists.
        for shards_list in self._dataset_filler_context.shard_lists.values():
            self._updated_infos.append(
                shards_list.write_config(
                    dataset_root_path=self._dataset.path,
                    hashes=self._dataset.dataset_structure.
                    hash_checksum_algorithms,
                ))

    def get_updated_infos(self) -> list[ShardListInfo]:
        """Return list of all ShardListInfo to update the dataset.
        """
        return self._updated_infos
