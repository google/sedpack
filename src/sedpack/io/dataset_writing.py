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
"""Mixin for sedpack.io.Dataset to do writing."""
import concurrent.futures
import contextlib
import itertools
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
)
import uuid

from tqdm.auto import tqdm

import sedpack
from sedpack.io.dataset_base import DatasetBase
from sedpack.io.file_info import FileInfo
from sedpack.io.merge_shard_infos import merge_splits_shard_infos
from sedpack.io.shard_file_metadata import ShardsList, ShardListInfo

import sedpack.io.utils as diutils
from sedpack.io.dataset_filler import DatasetFiller


class DatasetWriting(DatasetBase):
    """Mixin for sedpack.io.Dataset to do writing.
    """

    def write_multiprocessing(  # pylint: disable=too-many-arguments
        self,
        *,
        feed_writer: Callable[..., Any],
        custom_arguments: list[Any],
        custom_kwarguments: list[dict[str, Any]] | None = None,
        consistency_check: bool = True,
        single_process: bool = False,
        max_disk_utilization: float | None = 0.8,
    ) -> list[Any]:
        """Multiprocessing write support. Spawn `len(custom_arguments)`
        processes to write examples in parallel. Note that all computation is
        run on the CPU (using `tf.device("CPU")`) in order to prevent each
        process trying to allocate whole GPU device memory.

        Args:

            feed_writer (Callable[..., Any]): A function which gets a
            DatasetFiller and one of the argument from `custom_arguments`.
            Called with `dataset_filler` as the first positional argument,
            followed by `*args` from `custom_arguments` and `**kwargs` if also
            `custom_kwarguments` are provided (defaults to no keyword
            arguments).

            custom_arguments (list[Any]): A list of arguments which are passed
            to `feed_writer` instances. A pool of `len(custom_kwarguments)`
            processes is being created to do this.

            custom_kwarguments (list[dict[str, Any]] | None): A list of
            keyword arguments which are passed to `feed_writer` instances.
            Defaults to no keyword arguments. Needs to have the same length as
            `custom_arguments`.

            consistency_check (bool): Check consistency of files after
            everything is written. Checks hash checksums so might be slow.
            Defaults to `True`.

            single_process (bool): Debugging run in a single process. Defaults
            to False.

            max_disk_utilization (float | None): Maximum fraction of disk
            capacity in `(0.0, 1.0]` allowed before raising `DiskFullError`, or
            `None` to disable the disk utilization check. Defaults to `0.8`.

        Returns: A list of values returned by calls to `feed_writer`.

        Example use (see `docs/tutorials/writing/` for more information):
        `feed_writer` takes `dataset_filler: DatasetFiller` as its first
        positional argument, followed by arbitrary `*args` and `**kwargs`
        unpacked from the worker's entry in `custom_arguments` and
        `custom_kwarguments`:
        ```python
        def feed_worker(
            dataset_filler: DatasetFiller,
            chunk: list[ExampleT],
            split: SplitT = TRAIN_SPLIT,
        ) -> int:
            # First argument is always `dataset_filler`; remaining `*args` and
            # `**kwargs` come from `custom_arguments` and `custom_kwarguments`.
            with dataset_filler:
                for values in chunk:
                    dataset_filler.write_example(
                        values=values,
                        split=split,
                    )
            return len(chunk)

        counts = dataset.write_multiprocessing(
            feed_writer=feed_worker,
            custom_arguments=[(chunk_0,), (chunk_1,)],
            custom_kwarguments=[{"split": TRAIN_SPLIT}, {"split": TEST_SPLIT}],
        )
        ```
        Each worker receives a `DatasetFiller` isolated under a unique
        non-empty `relative_path_from_split` (so exiting the worker's `with`
        block updates metadata only up to
        `<split>/<relative_path_from_split>/shards_list.json`); once all
        workers finish, `write_multiprocessing` collects
        `dataset_filler.get_updated_infos()` and calls
        `self.write_config(updated_infos=updated_infos)`.
        """
        # Default to no kwargs.
        if custom_kwarguments is None:
            custom_kwarguments = [{} for _ in range(len(custom_arguments))]
        # Check the same length.
        if len(custom_arguments) != len(custom_kwarguments):
            raise ValueError(f"Custom args and kwargs lists have to have the "
                             f"same length when passed to "
                             f"`Dataset.write_multiprocessing` but have "
                             f"lengths: {len(custom_arguments) = } and "
                             f"{len(custom_kwarguments) = }")

        # Create dataset fillers (will be reused).
        dataset_fillers = [
            DatasetFiller(
                dataset=self,
                relative_path_from_split=Path(uuid.uuid4().hex),
                max_disk_utilization=max_disk_utilization,
            ) for _ in range(len(custom_arguments))
        ]

        # Wrapper inputs.
        wrapper_inputs = zip(
            itertools.repeat(feed_writer),
            dataset_fillers,
            custom_arguments,
            custom_kwarguments,
        )
        if single_process or not custom_arguments:
            # Testing mode.
            wrapper_outputs = list(map(_wrapper_func, wrapper_inputs))
        else:
            # Fill writers and collect results. Use ProcessPoolExecutor (whose
            # workers are non-daemon) so each DatasetFiller can use its own
            # ProcessPoolExecutor for shard writing.
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=len(custom_arguments)) as executor:
                wrapper_outputs = list(
                    executor.map(_wrapper_func, wrapper_inputs))

        # Retrieve filled dataset_fillers and feed_writer results.
        dataset_fillers = [
            dataset_filler for dataset_filler, _ in wrapper_outputs
        ]
        results = [result for _, result in wrapper_outputs]

        # Update
        updated_infos: list[ShardListInfo] = []
        for dataset_filler in dataset_fillers:
            # Beware that DatasetFiller objects were pickled and unpickled so
            # they no longer hold a reference of this object.
            updated_infos.extend(dataset_filler.get_updated_infos())
        self.write_config(updated_infos=updated_infos)

        # Check consistency of this dataset.
        if consistency_check:
            self.check()

        # Return user-defined results.
        return results

    def filler(
        self,
        concurrency: int = 4,
        max_disk_utilization: float | None = 0.8,
    ) -> DatasetFiller:
        """Return a dataset filler context manager for writing examples.

        Uses the default empty `relative_path_from_split=Path(".")`, so exiting
        the `with` block automatically updates `<split>/shards_list.json` and
        the root `dataset_info.json`. For multi-worker or distributed writing,
        use `Dataset.write_multiprocessing` or `DatasetFiller` with a non-empty
        `relative_path_from_split` (see `DatasetFiller` docstring).

        Args:

          concurrency (int): Setting to a positive integer allows writing shard
          files in parallel. Defaults to 4.

          max_disk_utilization (float | None): Maximum fraction of disk capacity
          in `(0.0, 1.0]` allowed before raising `DiskFullError`, or `None` to
          disable the disk utilization check. Defaults to `0.8` (80%). Checked
          at creation (`__init__`) and in `write_example` before opening a new
          shard. When `DiskFullError` is raised during `write_example`,
          `DatasetFiller.__exit__` still flushes and saves all so-far opened
          shards so no previously written data is lost (which may cause final
          disk utilization to slightly exceed `max_disk_utilization`).

        Example use:
        # Context manager properly opens and closes shards.
        with dataset.filler() as dataset_filler:
            # Add examples, new shards are opened automatically.
            for values in examples:
                dataset_filler.write_example(
                    values=values,
                    split=split,
                )
        """
        return DatasetFiller(
            dataset=self,
            concurrency=concurrency,
            max_disk_utilization=max_disk_utilization,
        )

    def write_config(
        self,
        updated_infos: list[ShardListInfo] | None = None,
    ) -> FileInfo:
        """Save configuration as json.

        When multiple `DatasetFiller` instances are used with a non-empty
        `relative_path_from_split` (see `DatasetFiller` docstring for the
        advanced use-case and caveats), pass the concatenated
        `filler.get_updated_infos()` list here after all fillers have exited.

        Args:

          updated_infos (list[ShardListInfo] | None): List of updates or new
          `ShardListInfo` descriptors (as returned by
          `DatasetFiller.get_updated_infos()`). Each `ShardListInfo` references
          a written `shards_list.json` file (such as
          `<split>/<relative_path_from_split>/shards_list.json` or a leaf
          `shards_list.json`, with `file_path` relative to `dataset.path`)
          along with its `hash_checksums`, `number_of_examples`, and
          `number_of_shards`. `write_config` recursively merges these shard
          lists up through intermediate directories to
          `<split>/shards_list.json` and writes the root `dataset_info.json`.

        Returns: a `FileInfo` with the path and hash checksums corresponding to
        the root `dataset_info.json` file.
        """
        if updated_infos:
            self._dataset_info.splits.update(
                merge_splits_shard_infos(
                    updates=updated_infos,
                    dataset_root=self.path,
                    hashes=self.dataset_structure.hash_checksum_algorithms,
                ))

        # Update the main dataset info config.
        return diutils.safe_update_file(
            dataset_root_path=self.path,
            relative_path=DatasetWriting._get_config_path(self.path,
                                                          relative=True),
            # Always serialize licenses (even when default).
            info=self._dataset_info.model_dump_json(indent=2),
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )

    def _check_shard_list_info(self, shard_list_info: ShardListInfo) -> None:
        """Recursively check consistency of a `ShardsList` file and its
        children.

        Verifies the file's hash checksums, `relative_path_self`,
        `number_of_examples`, and `number_of_shards` against its `ShardListInfo`
        descriptor and child entries.

        Args:

          shard_list_info (ShardListInfo): Descriptor of the `shards_list.json`
          file to validate.

        Raises: ValueError if any checksum, path, or count is inconsistent.
        """
        # Check the current file.
        file_path: Path = shard_list_info.shard_list_info_file.file_path
        real: tuple[str, ...] = sedpack.io.utils.hash_checksums(
            file_path=self.path / file_path,
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )
        expected: tuple[
            str, ...] = shard_list_info.shard_list_info_file.hash_checksums
        if real != expected:
            raise ValueError(f"Hash checksum miss-match in {file_path} "
                             f"{real = } {expected = }")

        # Check children and count consistency.
        shard_list: ShardsList = ShardsList.model_validate_json(
            (self.path / file_path).read_text())
        if shard_list.relative_path_self != file_path:
            raise ValueError(
                f"Path mismatch in {file_path}: "
                f"{shard_list.relative_path_self = } != {file_path = }")

        expected_examples: int = (
            sum(s.number_of_examples for s in shard_list.shard_files) +
            sum(c.number_of_examples for c in shard_list.children_shard_lists))
        if (shard_list.number_of_examples != expected_examples or
                shard_list_info.number_of_examples != expected_examples):
            raise ValueError(f"Number of examples mismatch in {file_path}: "
                             f"{shard_list.number_of_examples = }, "
                             f"{shard_list_info.number_of_examples = }, "
                             f"{expected_examples = }")

        expected_shards: int = len(shard_list.shard_files) + sum(
            c.number_of_shards for c in shard_list.children_shard_lists)
        if shard_list_info.number_of_shards != expected_shards:
            raise ValueError(f"Number of shards mismatch in {file_path}: "
                             f"{shard_list_info.number_of_shards = }, "
                             f"{expected_shards = }")

        for child in shard_list.children_shard_lists:
            self._check_shard_list_info(child)

    def current_metadata_checksums(self) -> tuple[str, ...]:
        """Return the hash checksums of the current version of
        `dataset_info.json`. Make sure to save using `write_config` before
        making a note of these values.
        """
        file_path: Path = self._get_config_path(self.path)
        return sedpack.io.utils.hash_checksums(
            file_path=file_path,
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )

    def check(
        self,
        show_progressbar: bool = True,
        hash_checksums_values: tuple[str, ...] = ()
    ) -> None:
        """Check the dataset integrity.

        Args:

          show_progressbar: Use tqdm to show a progressbar for different checks.

          hash_checksums_values (tuple[str, ...]): Expected hash sum values of
          the main metadata file.

        Raises: ValueError if the dataset is inconsistent.
        """
        # Check the main metadata file.
        if hash_checksums_values:
            real: tuple[str, ...] = self.current_metadata_checksums()
            if real != hash_checksums_values:
                file_path: Path = self._get_config_path(self.path)
                raise ValueError(f"Hash checksum miss-match in {file_path}")

        # Check metadata consistency.
        for shard_list_info in self._dataset_info.splits.values():
            self._check_shard_list_info(shard_list_info)

        # Check shards consistency.
        for split in self._dataset_info.splits:
            for shard_info in tqdm(list(self.shard_info_iterator(split)),
                                   desc=f"Validating checksums for {split}",
                                   disable=not show_progressbar):
                for file_info in shard_info.file_infos:
                    real_hashes = sedpack.io.utils.hash_checksums(
                        file_path=self.path / file_info.file_path,
                        hashes=self._dataset_info.dataset_structure.
                        hash_checksum_algorithms)
                    if real_hashes != file_info.hash_checksums:
                        raise ValueError(
                            f"Hash checksum miss-match in {file_info.file_path}"
                            f" got: {real_hashes} but expected "
                            f"{file_info.hash_checksums}")


# We want to get results back.
def _wrapper_func(
    feed_writer_dataset_filler_args_kwargs: tuple[Callable[...,
                                                           Any], DatasetFiller,
                                                  Any, Mapping[str, Any]]
) -> tuple[DatasetFiller, Any]:
    """Helper function for write_multiprocessing. Needs to be pickleable.
    """
    # Prevent each process from hoarding the whole GPU memory.
    context: contextlib.AbstractContextManager[Any]
    try:
        # TensorFlow is an optional dependency.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        context = tf.device("CPU")
    except ImportError:
        context = contextlib.nullcontext()
    with context:
        (
            feed_writer,
            dataset_filler,
            args,
            kwargs,
        ) = feed_writer_dataset_filler_args_kwargs
        result = feed_writer(dataset_filler, *args, **kwargs)
        return (dataset_filler, result)
