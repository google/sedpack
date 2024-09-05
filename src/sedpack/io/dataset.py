# Copyright 2023-2024 Google LLC
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
"""Build and load tensorFlow dataset Record wrapper"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union
import os
import uuid

import asyncstdlib
from tqdm.auto import tqdm
import semver
import tensorflow as tf

import sedpack
from sedpack.io.errors import DatasetExistsError
from sedpack.io.itertools import round_robin, round_robin_async, shuffle_buffer
from sedpack.io.itertools import LazyPool
from sedpack.io.metadata import DatasetInfo, DatasetStructure, Metadata
from sedpack.io.merge_shard_infos import merge_shard_infos
from sedpack.io.shard_file_metadata import ShardInfo, ShardsList, ShardListInfo
from sedpack.io.shard import IterateShardBase
from sedpack.io.shard.iterate_shard_base import T
from sedpack.io.flatbuffer import IterateShardFlatBuffer
from sedpack.io.npz import IterateShardNP
from sedpack.io.tfrec import IterateShardTFRec
from sedpack.io.tfrec.tfdata import get_from_tfrecord

from sedpack.io.types import ExampleT, ShardFileTypeT, SplitT, TFDatasetT
import sedpack.io.utils as diutils
from sedpack.io.dataset_filler import DatasetFiller


class Dataset:
    """Dataset class."""

    def __init__(self,
                 path: Union[str, Path],
                 create_dataset: bool = False) -> None:
        """Class for saving and loading a database.

        Args:

            path (Union[str, Path]): Path from where to load the dataset (a
            directory -- for instance
            "/home/user_name/datasets/my_awesome_dataset").

            create_dataset (bool): Are we creating a new dataset? Defaults to
            False which is used when (down-)loading a dataset.

        Raises:

            ValueError if the dataset was created using a newer version of the
            sedpack than the one trying to load it. See
            sedpack.__version__ docstring.

            FileNotFoundError if `create_dataset` is False and the
            `dataset_info.json` file does not exist.
        """
        # Ensure pathlib Path.
        self.path: Path = Path(path)
        try:
            # Expand user directory.
            self.path = self.path.expanduser()
        except RuntimeError:
            # Expansion failed we assume that the path was not supposed to be
            # expanded.
            pass

        # Resolve the dataset root path to avoid problems when checking if
        # another path is relative to it.
        self.path = self.path.resolve()

        # Default DatasetInfo.
        self._dataset_info = DatasetInfo()

        if not create_dataset:
            # Load the information.
            self._dataset_info = Dataset._load(self.path)

        self._logger = logging.getLogger("sedpack.io.Dataset")

    @staticmethod
    def _load(path: Path) -> DatasetInfo:
        """Load a dataset from a config file.

        Args:

            path (Path): The path to the dataset.

        Raises:

            ValueError if the dataset was created using a newer version of the
            sedpack than the one trying to load it. See
            sedpack.__version__ docstring.

            FileNotFoundError if the `dataset_info.json` file does not exist.

        Returns: A DatasetInfo object representing the dataset metadata.
        """
        dataset_info = DatasetInfo.model_validate_json(
            Dataset._get_config_path(path).read_text(encoding="utf-8"))

        # Check that the library version (version of this software) is not
        # lower than what was used to capture the dataset.
        if semver.Version.parse(dataset_info.metadata.sedpack_version).compare(
                sedpack.__version__) > 0:
            raise ValueError(f"Dataset-lib module is outdated, "
                             f"sedpack_version: {sedpack.__version__}, "
                             f"but dataset was created using: "
                             f"{dataset_info.metadata.sedpack_version}")

        return dataset_info

    @staticmethod
    def create(
        path: Union[str, Path],
        metadata: Metadata,
        dataset_structure: DatasetStructure,
    ) -> "Dataset":
        """Create an empty dataset to be filled using the `filler` or
        `write_multiprocessing` API.

        Args:

            path (Union[str, Path]): Path where the dataset gets saved (a
            directory -- for instance
            "/home/user_name/datasets/my_awesome_dataset").

            metadata (Metadata): Information about this dataset.

            dataset_structure (DatasetStructure): Structure of saved records.

        Raises: DatasetExistsError if creating this object would overwrite the
        corresponding config file.
        """
        # Create a new object.
        dataset = Dataset(path=path, create_dataset=True)

        # Do not overwrite an existing dataset.
        if Dataset._get_config_path(dataset.path).is_file():
            # Raise if the dataset already exists.
            raise DatasetExistsError(dataset.path)

        # Create a new dataset directory if needed.
        dataset.path.mkdir(parents=True, exist_ok=True)

        # Fill metadata and structure parameters.
        dataset.metadata = metadata
        dataset.dataset_structure = dataset_structure

        # Write empty config.
        dataset.write_config(updated_infos=[])
        return dataset

    @property
    def metadata(self) -> Metadata:
        return self._dataset_info.metadata

    @metadata.setter
    def metadata(self, value: Metadata) -> None:
        self._dataset_info.metadata = value

    @property
    def dataset_structure(self) -> DatasetStructure:
        return self._dataset_info.dataset_structure

    @dataset_structure.setter
    def dataset_structure(self, value: DatasetStructure) -> None:
        self._dataset_info.dataset_structure = value

    def _shard_info_iterator(
            self, shard_list_info: ShardListInfo) -> Iterator[ShardInfo]:
        """Recursively yield `ShardInfo` from the whole directory tree.
        """
        shard_list: ShardsList = ShardsList.model_validate_json(
            (self.path /
             shard_list_info.shard_list_info_file.file_path).read_text())

        yield from shard_list.shard_files

        for child in shard_list.children_shard_lists:
            yield from self._shard_info_iterator(child)

    def shard_info_iterator(self, split: SplitT) -> Iterator[ShardInfo]:
        """Iterate all `ShardInfo` in the split.

        Args:

          split (SplitT): Which split to iterate shard information from.

        Raises: ValueError when the split is not present. A split not being
        present is different from there not being any shard.
        """
        if split not in self._dataset_info.splits:
            # Split not present.
            raise ValueError(f"There is no shard in {split}.")

        shard_list_info: ShardListInfo = self._dataset_info.splits[split]

        yield from self._shard_info_iterator(shard_list_info)

    def shard_paths_dataset(
        self,
        split: SplitT,
        shards: Optional[int] = None,
        shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
    ) -> list[str]:
        """Return a list of shard filenames.

        Args:

            split (SplitT): Split, see SplitT.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

        Returns: A list of shards filenames.
        """
        # List of all shard informations
        shards_list: list[ShardInfo] = list(
            self.shard_info_iterator(split=split))

        # Filter which shards to use.
        if shard_filter is not None:
            shards_list = list(filter(shard_filter, shards_list))

            kept_metadata: set[str] = {
                str(s.custom_metadata) for s in shards_list
            }
            self._logger.info(
                "Filtered shards with custom metadata: %s from split: %s",
                kept_metadata,
                split,
            )

        # Check that there is still something to iterate
        if not shards_list:
            raise ValueError("The list of shards is empty. Try less "
                             "restrictive filtering.")

        # Truncate the shard list
        if shards:
            shards_list = shards_list[:shards]

        # Full shard file paths.
        shard_paths = [
            # TODO allow shards consisting of multiple files
            str(self.path / s.file_infos[0].file_path) for s in shards_list
        ]

        return shard_paths

    def read_and_decode(self, tf_dataset: TFDatasetT,
                        cycle_length: Optional[int],
                        num_parallel_calls: Optional[int],
                        parallelism: Optional[int]) -> TFDatasetT:
        """Read the shard files and decode them.

        Args:

          tf_dataset (tf.data.Dataset): Dataet containing shard paths as
          strings.

          cycle_length (Optional[int]): How many files to read at once.

          num_parallel_calls (Optional[int]): Number of parallel reading calls.

          parallelism (Optional[int]): Decoding parallelism.

        Returns: tf.data.Dataset containing decoded examples.
        """
        # If the cycle_length is None it is determined automatically and we do
        # not need determinism. See documentation
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
        deterministic: Optional[bool] = True
        if isinstance(cycle_length, int):
            deterministic = False if cycle_length > 1 else None
        elif cycle_length is None:
            deterministic = True

        # TODO test: TFRecordDataset takes a list of files (we are passing a
        # single file now). Should we batch here for improved performance?

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the faster and more compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        #
        # If shuffle is equal to zero we produce deterministic order of
        # examples. By setting cycle_length to one (and num_parallel_calls to
        # default) we also avoid non-obvious interleaving patterns when there
        # are only a few shards. Deterministic defaults in order to avoid a
        # warning (deterministic = False does nothing when there is no thread
        # pool created by num_parallel_calls).
        tf_dataset = tf_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(
                x, compression_type=self.dataset_structure.compression),
            cycle_length=cycle_length,
            block_length=1,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
        )

        # Decode to records
        tf_dataset = tf_dataset.map(
            get_from_tfrecord(self.dataset_structure.saved_data_description),
            num_parallel_calls=parallelism,
        )

        return tf_dataset

    def as_tfdataset(self,
                     split: SplitT,
                     process_record: Optional[Callable[[ExampleT], T]] = None,
                     shards: Optional[int] = None,
                     shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
                     repeat: bool = True,
                     batch_size: int = 32,
                     prefetch: int = 2,
                     file_parallelism: Optional[int] = os.cpu_count(),
                     parallelism: Optional[int] = os.cpu_count(),
                     shuffle: int = 1_000) -> TFDatasetT:
        """"Dataset as tfdataset

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Optional[Callable[[ExampleT], T]]): Optional
            function that processes a single record.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            batch_size (int): Number of examples in a single batch. No batching
            if the `batch_size` is less or equal to zero.

            prefetch (int): Prefetch this many batches.

            file_parallelism (Optional[int]): IO parallelism.

            parallelism (Optional[int]): Parallelism of trace decoding and
            processing (ignored if shuffle is zero).

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: A tf.data.Dataset object of infinite stream of shuffled and
        batched examples.
        """
        # Shard file names.
        shard_paths: list[str] = self.shard_paths_dataset(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
        )

        if self.dataset_structure.shard_file_type != "tfrec":
            raise ValueError(f"The method as_tfdataset is only supported for "
                             f"tfrec but "
                             f"{self.dataset_structure.shard_file_type} was "
                             f"provided.")

        # Dataset creation
        tf_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)

        # Infinite loop over the shard paths
        if repeat:
            tf_dataset = tf_dataset.repeat()

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            tf_dataset = tf_dataset.shuffle(len(shard_paths))

        tf_dataset = self.read_and_decode(
            tf_dataset=tf_dataset,
            cycle_length=file_parallelism if shuffle else 1,
            num_parallel_calls=file_parallelism if shuffle else None,
            parallelism=parallelism,
        )

        # Process each record if requested
        if process_record:
            tf_dataset = tf_dataset.map(process_record,
                                        num_parallel_calls=parallelism)

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            tf_dataset = tf_dataset.shuffle(shuffle)

        if batch_size > 0:
            # Batch
            tf_dataset = tf_dataset.batch(batch_size)

        # Prefetch (to keep loading when GPU is processing).
        tf_dataset = tf_dataset.prefetch(prefetch)

        return tf_dataset

    def check(self, show_progressbar: bool = True) -> bool:
        """Check the dataset integrity.

        Args:
          show_progressbar: Use tqdm to show a progressbar for different checks.

        Raises: ValueError if the dataset is inconsistent.

        Returns: True if all hashes match, raises a ValueError otherwise.
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
        return True

    def write_config(self,
                     updated_infos: list[ShardListInfo]) -> tuple[str, ...]:
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
            splits_to_update[split].append(info)

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
            relative_path=Dataset._get_config_path(self.path, relative=True),
            # Always serialize licenses (even when default).
            info=self._dataset_info.model_dump_json(indent=2),
            hashes=self.dataset_structure.hash_checksum_algorithms,
        )

    @staticmethod
    def _get_config_path(path: Path, relative: bool = False) -> Path:
        """Get the dataset config path.

        Args:

          path (Path): The dataset path.

          relative (bool): Return only relative to `path`. Defaults to False.

        Returns: A path to the config.
        """
        relative_path: Path = Path("dataset_info.json")
        if relative:
            return relative_path
        else:
            return path / relative_path

    def write_multiprocessing(self,
                              feed_writer: Callable[..., Any],
                              custom_arguments: List[Any],
                              custom_kwarguments: Optional[List[Dict[
                                  str, Any]]] = None,
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
        wrapper_inputs = zip(itertools.repeat(feed_writer), dataset_fillers,
                             custom_arguments, custom_kwarguments)
        if single_process:
            # Testing mode.
            wrapper_outputs = list(map(_wrapper_func, wrapper_inputs))
        else:
            # Fill writers and collect results.
            # TODO consider capping the number of processes by CPU count or a
            # parameter.
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
            # TODO implement multiprocessing check.
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

    async def as_numpy_iterator_async(
        self,
        split: SplitT,
        process_record: Optional[Callable[[ExampleT], T]] = None,
        shards: Optional[int] = None,
        shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
        repeat: bool = True,
        file_parallelism: int = os.cpu_count(),
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation. Iterates in random order.

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Optional[Callable[[ExampleT], T]]): Optional
            function that processes a single record.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            file_parallelism (int): IO parallelism.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self._as_numpy_common(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decode the files.
        supported_file_types: list[ShardFileTypeT] = [
            "npz",
            "fb",
        ]
        if self.dataset_structure.shard_file_type not in supported_file_types:
            raise ValueError(f"The method as_numpy_iterator_async supports "
                             f"only {supported_file_types} but not "
                             f"{self.dataset_structure.shard_file_type}")

        shard_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case _:
                raise ValueError(f"{self.dataset_structure.shard_file_type} "
                                 f"is marked as supported but support is not "
                                 f"implemented.")

        # Automatically shuffle.
        if shuffle:
            example_iterator = round_robin_async(
                asyncstdlib.map(shard_iterator.iterate_shard_async,
                                shard_paths_iterator),
                buffer_size=file_parallelism,
            )
        else:
            example_iterator = asyncstdlib.chain.from_iterable(
                asyncstdlib.map(shard_iterator.iterate_shard_async,
                                shard_paths_iterator))

        # Process each record if requested.
        if process_record:
            example_iterator = asyncstdlib.map(process_record, example_iterator)

        async for example in example_iterator:
            yield example

    def _as_numpy_common(
        self,
        split: SplitT,
        shards: Optional[int] = None,
        shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
        repeat: bool = True,
        shuffle: int = 1_000,
    ) -> Iterable[str]:
        """"Common part of as_numpy_iterator and as_numpy_iterator_*.

        Args:

            split (SplitT): Split, see SplitT.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over shard paths.
        """
        # Shard file names.
        shard_paths: list[str] = self.shard_paths_dataset(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
        )

        # Infinite loop over the shard paths
        if repeat:
            shard_paths_iterator = itertools.cycle(shard_paths)
        else:
            shard_paths_iterator = shard_paths

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            shard_paths_iterator = shuffle_buffer(shard_paths_iterator,
                                                  buffer_size=len(shard_paths))
        return shard_paths_iterator

    def as_numpy_iterator_concurrent(
        self,
        split: SplitT,
        process_record: Optional[Callable[[ExampleT], T]] = None,
        shards: Optional[int] = None,
        shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
        repeat: bool = True,
        file_parallelism: int = os.cpu_count() or 1,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation. Iterates in random order.

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Optional[Callable[[ExampleT], T]]): Optional
            function that processes a single record.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            file_parallelism (int): IO parallelism. Defaults to `os.cpu_count()
            or 1`.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self._as_numpy_common(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decoding and processing function based on shard file type.
        if process_record:
            shard_iterator: IterateShardBase[T]
        else:
            shard_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "tfrec":
                shard_iterator = IterateShardTFRec(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                    num_parallel_calls=1,
                )
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=process_record,
                )
            case _:
                raise ValueError("Unsupported shard_file_type "
                                 f"{self.dataset_structure.shard_file_type}")

        with tf.device("CPU"):
            if shuffle:
                with LazyPool(file_parallelism) as pool:
                    yield from round_robin(
                        pool.imap_unordered(
                            shard_iterator.process_and_list,
                            shard_paths_iterator,
                        ),
                        # round_robin keeps the whole shard files in memory.
                        buffer_size=file_parallelism,
                    )
            else:
                # Iterate in order but avoid out of memory caused by eagerly
                # mapping a reading function across all shards.
                with ThreadPoolExecutor(
                        max_workers=file_parallelism) as executor:
                    # Do not iterate the start multiple times.
                    shard_paths_iterator = iter(shard_paths_iterator)
                    batch = list(
                        itertools.islice(shard_paths_iterator,
                                         file_parallelism))
                    while batch:
                        yield from itertools.chain.from_iterable(
                            executor.map(shard_iterator.process_and_list,
                                         batch))
                        batch = list(
                            itertools.islice(shard_paths_iterator,
                                             file_parallelism))

    def as_numpy_iterator(
        self,
        split: SplitT,
        process_record: Optional[Callable[[ExampleT], T]] = None,
        shards: Optional[int] = None,
        shard_filter: Optional[Callable[[ShardInfo], bool]] = None,
        repeat: bool = True,
        shuffle: int = 1_000,
    ) -> Iterable[ExampleT] | Iterable[T]:
        """"Dataset as a numpy iterator (no batching). Pure Python
        implementation.

        Args:

            split (SplitT): Split, see SplitT.

            process_record (Optional[Callable[[ExampleT], T]]): Optional
            function that processes a single record.

            shards (Optional[int]): If specified limits the dataset to the
            first `shards` shards.

            shard_filter (Optional[Callable[[ShardInfo], bool]): If present
            this is a function taking the ShardInfo and returning True if the
            shard shall be used for traversal and False otherwise.

            repeat (bool): Whether to repeat examples and thus create infinite
            dataset.

            shuffle (int): How many examples should be shuffled across shards.
            When set to 0 the iteration is deterministic. It might be faster to
            iterate over shuffled dataset.

        Returns: An iterator over numpy examples (unless the parameter
        `process_record` returns something else). No batching is done.
        """
        shard_paths_iterator: Iterable[str] = self._as_numpy_common(
            split=split,
            shards=shards,
            shard_filter=shard_filter,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Decode the files.
        shard_iterator: IterateShardBase[ExampleT]
        match self.dataset_structure.shard_file_type:
            case "tfrec":
                shard_iterator = IterateShardTFRec(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                    num_parallel_calls=os.cpu_count() or 1,
                )
            case "npz":
                shard_iterator = IterateShardNP(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case "fb":
                shard_iterator = IterateShardFlatBuffer(
                    dataset_structure=self.dataset_structure,
                    process_record=None,
                )
            case _:
                raise ValueError(f"Unknown shard_file_type "
                                 f"{self.dataset_structure.shard_file_type}")

        example_iterator = itertools.chain.from_iterable(
            map(shard_iterator.iterate_shard, shard_paths_iterator))

        # Process each record if requested
        if process_record:
            example_iterator = map(process_record, example_iterator)

        # Randomize only if > 0 -- no shuffle in test/validation
        if shuffle:
            example_iterator = shuffle_buffer(example_iterator,
                                              buffer_size=shuffle)

        yield from example_iterator


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
