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
from collections import defaultdict
import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    get_args,
)
import uuid

from tqdm.auto import tqdm
import tensorflow as tf

import sedpack
from sedpack.io.dataset_base import DatasetBase
from sedpack.io.file_info import FileInfo
from sedpack.io.merge_shard_infos import merge_shard_infos
from sedpack.io.shard_file_metadata import ShardListInfo

from sedpack.io.types import SplitT
import sedpack.io.utils as diutils
from sedpack.io.dataset_filler import DatasetFiller


class DatasetWriting(DatasetBase):
    """Mixin for sedpack.io.Dataset to do writing.
    """

    def write_multiprocessing(  # pylint: disable=too-many-arguments
            self,
            *,
            feed_writer: Callable[..., Any],
            custom_arguments: List[Any],
            custom_kwarguments: Optional[List[Dict[str, Any]]] = None,
            consistency_check: bool = True,
            single_process: bool = False) -> List[Any]:
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

            custom_arguments (List[Any]): A list of arguments which are passed
            to `feed_writer` instances. A pool of `len(custom_kwarguments)`
            processes is being created to do this.

            custom_kwarguments (Optional[List[Dict[str, Any]]]): A list of
            keyword arguments which are passed to `feed_writer` instances.
            Defaults to no keyword arguments. Needs to have the same length as
            `custom_arguments`.

            consistency_check (bool): Check consistency of files after
            everything is written. Checks hash checksums so might be slow.
            Defaults to `True`.

            single_process (bool): Debugging run in a single process. Defaults
            to False.

        Returns: A list of values returned by calls to `feed_writer`.

        Example use: See docs/tutorials/writing/ for more information.
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
                auto_update_dataset=False,
            ) for _ in range(len(custom_arguments))
        ]

        # Wrapper inputs.
        wrapper_inputs = zip(
            itertools.repeat(feed_writer),
            dataset_fillers,
            custom_arguments,
            custom_kwarguments,
        )
        if single_process:
            # Testing mode.
            wrapper_outputs = list(map(_wrapper_func, wrapper_inputs))
        else:
            # Fill writers and collect results.
            with Pool(len(custom_arguments)) as pool:
                wrapper_outputs = list(pool.imap(_wrapper_func, wrapper_inputs))

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

    def filler(self) -> DatasetFiller:
        """Return a dataset filler context manager for writing examples.

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
        return DatasetFiller(self)

    def write_config(self, updated_infos: list[ShardListInfo]) -> FileInfo:
        """Save configuration as json.

        Args:

          updated_infos (list[ShardListInfo]): List of updates or new shard
          list infos.

        Returns: a tuple of hash checksums corresponding to the root dataset
        info file.
        """
        # Update shards lists in splits.
        splits_to_update: defaultdict[SplitT,
                                      list[ShardListInfo]] = defaultdict(list)

        # Sort the infos by their split.
        for info in updated_infos:
            # The path is at least `split / shards_list.json` or longer.
            split = str(info.shard_list_info_file.file_path.parts[0])
            # Type narrowing with get_args does not seem to work with mypy.
            if split not in get_args(SplitT):
                raise ValueError(f"Not a known split value: {split}")
            splits_to_update[split].append(info)  # type: ignore

        # Merge recursively.
        for split, updates in splits_to_update.items():
            self._dataset_info.splits[split] = merge_shard_infos(
                updates=updates,
                dataset_root=self.path,
                common=1,
                hashes=self.dataset_structure.hash_checksum_algorithms,
            )

        # Update the main dataset info config.
        return diutils.safe_update_file(
            dataset_root_path=self.path,
            relative_path=DatasetWriting._get_config_path(self.path,
                                                          relative=True),
            # Always serialize licenses (even when default).
            info=self._dataset_info.model_dump_json(indent=2),
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )

    def check(self, show_progressbar: bool = True) -> None:
        """Check the dataset integrity.

        Args:
          show_progressbar: Use tqdm to show a progressbar for different checks.

        Raises: ValueError if the dataset is inconsistent.
        """
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
                        )


# We want to get results back.
def _wrapper_func(
    feed_writer_dataset_filler_args_kwargs: Tuple[Callable[...,
                                                           Any], DatasetFiller,
                                                  Any, Mapping[str, Any]]
) -> Tuple[DatasetFiller, Any]:
    """Helper function for write_multiprocessing. Needs to be pickleable.
    """
    # Prevent each process from hoarding the whole GPU memory.
    with tf.device("CPU"):
        (
            feed_writer,
            dataset_filler,
            args,
            kwargs,
        ) = feed_writer_dataset_filler_args_kwargs
        result = feed_writer(dataset_filler, *args, **kwargs)
        return (dataset_filler, result)
