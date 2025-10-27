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
"""The default shard info iterator."""
from collections import defaultdict
from collections.abc import Hashable
import heapq
import itertools
import json
import logging
from pathlib import Path

from typing import Callable, Iterator

from sedpack.io.metadata import DatasetInfo
from sedpack.io.shard_file_metadata import ShardInfo
from sedpack.io.shard_info_iterator.shard_info_iterator import ShardInfoIterator
from sedpack.io.types import SplitT
from sedpack.io.itertools import shuffle_buffer


class _SingleLevelBalancer:
    """Take a bunch of iterators of `ShardInfo` and interleave them such that
    the number of seen examples from each of them is roughly the same (or
    weighted). When one or more of the iterators are exhausted continue until
    all of them are exhausted.
    """

    def __init__(
        self,
        iterators: list[Iterator[tuple[float, ShardInfo]]],
    ) -> None:
        """Initialize the balancing.

        Args:

          iterators (list[Iterator[tuple[float, ShardInfo]]]): The iterators to
          be interleaved fairly. The float is interpreted as the `weight`.
          Meaning each example counts for `weight`.
        """
        self.iterators: list[Iterator[tuple[float, ShardInfo]]] = [
            iter(i) for i in iterators
        ]
        self.balancing_heap: list[tuple[float, int]] = [
            # (weighted seen examples, id of the iterator)
            (0.0, i) for i in range(len(iterators))
        ]

    def __iter__(self) -> Iterator[ShardInfo]:
        """Return the shard information iterator (reentrant).
        """
        return self

    def __next__(self) -> ShardInfo:
        """Return the next `ShardInfo` and the corresponding weight.
        """
        while self.balancing_heap:
            seen_examples, iterator_id = heapq.heappop(self.balancing_heap)
            try:
                weight, shard_info = next(self.iterators[iterator_id])
                heapq.heappush(
                    self.balancing_heap,
                    (
                        seen_examples +
                        (weight * shard_info.number_of_examples),
                        iterator_id,
                    ),
                )
                return shard_info
            except StopIteration:
                pass

        raise StopIteration


def _split_balancing(
    shard_list: list[ShardInfo],
    balance_by: tuple[Callable[[ShardInfo], Hashable], ...],
    repeat: bool,
    shuffle: int,
) -> Iterator[ShardInfo]:
    """Balance in a specified order.

    Args:

      shard_list (list[ShardInfo]): The list of shards to be balanced.

      balance_by (tuple[Callable[[ShardInfo], Hashable], ...]): The list of
      priority of balancing. The first will be the most important to be
      balanced. If this callable is an object with a `weight(self, shard_info)
      -> float` method then each example from this shard counts for `weight`.
      Otherwise each example counts as 1. Meaning that setting the weight to
      0.5 will result into seeing twice as many of these shards. Be careful
      with weights of zero and negative.

      repeat (bool): Should the `ShardInfo` be repeated indefinitely?

      shuffle (int): The size of shuffle buffer in the lowest level iteration.

    Returns: an iterator of the `ShardInfo` objects.
    """
    if not balance_by:
        inner_iterator: Iterator[ShardInfo]
        if repeat:
            inner_iterator = itertools.cycle(shard_list)
        else:
            inner_iterator = iter(shard_list)
        if shuffle:
            inner_iterator = iter(
                shuffle_buffer(
                    iterable=inner_iterator,
                    buffer_size=shuffle,
                ))
        return inner_iterator

    classes: defaultdict[Hashable, list[ShardInfo]] = defaultdict(list)
    current_balancer: Callable[[ShardInfo], Hashable] = balance_by[0]

    for shard_info in shard_list:
        classes[current_balancer(shard_info)].append(shard_info)

    iterators: list[Iterator[ShardInfo]] = [
        _split_balancing(
            shard_list=v,
            balance_by=balance_by[1:],
            repeat=repeat,
            shuffle=shuffle,
        ) for v in classes.values()
    ]

    # How do we get weights from the current balancer.
    if (hasattr(current_balancer, "weight") and
            callable(current_balancer.weight)):

        def prepend_weight(shard_info: ShardInfo) -> tuple[float, ShardInfo]:
            return (
                current_balancer.weight(shard_info),
                shard_info,
            )
    else:

        def prepend_weight(shard_info: ShardInfo) -> tuple[float, ShardInfo]:
            return (
                1.0,  # Default just count examples.
                shard_info,
            )

    return _SingleLevelBalancer(
        iterators=[map(prepend_weight, i) for i in iterators])


class BalancedShardInfoIterator(ShardInfoIterator):
    """Iterate shards of a dataset.
    """

    def __init__(
            self,
            *,
            dataset_path: Path,
            dataset_info: DatasetInfo,
            split: SplitT | None,
            repeat: bool = True,
            shard_filter: Callable[[ShardInfo], bool] | None = None,
            shuffle: int = 0,
            balance_by: tuple[Callable[[ShardInfo], Hashable], ...] = (),
    ) -> None:
        """Initialize shard information iteration.

        Args:

          dataset_path (Path): The path to the dataset directory.

          dataset_info (DatasetInfo): The information about the iterated
          dataset.

          split (SplitT | None): Which split to iterate or all if set to None.

          repeat (bool): Should we cycle indefinitely? You most likely want to
          set this to True especially when using `balance_by` since otherwise
          the beginning will be balanced but whenever one type of shards will
          be less prevalent it will not appear towards the end.

          shard_filter (Callable[[ShardInfo], bool] | None): If present this is
          a function taking the ShardInfo and returning True if the shard shall
          be used for traversal and False otherwise.

          shuffle (int): When set to 0 the iteration is deterministic otherwise
          shuffle the shards with a shuffle buffer of at least `shuffle`
          elements. Current implementation shuffles all shard information.

          balance_by (tuple[Callable[[ShardInfo], Hashable], ...]): The list of
          priority of balancing. The first will be the most important to be
          balanced. If this callable is an object with a `weight(self,
          shard_info) -> float` method then each example from this shard counts
          for `weight`.  Otherwise each example counts as 1. Meaning that
          setting the weight to 0.5 will result into seeing twice as many of
          these shards. Be careful with weights of zero and negative.
        """
        super().__init__(
            dataset_path=dataset_path,
            dataset_info=dataset_info,
            split=split,
            repeat=repeat,
        )

        self.shuffle: int = shuffle

        # Logging for non-trivial operations such as filtering custom metadata.
        self._logger = logging.getLogger(__name__)

        # Cache the list of shards.
        shard_list: list[ShardInfo] = list(
            ShardInfoIterator(
                dataset_path=dataset_path,
                dataset_info=dataset_info,
                split=split,
                repeat=False,
            ))

        # Filter if needed.
        if shard_filter:
            shard_list = [
                shard_info for shard_info in shard_list
                if shard_filter(shard_info)
            ]

            kept_metadata: set[str] = {
                json.dumps(
                    s.custom_metadata,
                    sort_keys=True,
                ) for s in shard_list
            }
            self._logger.info(
                "Filtered shards with custom metadata: %s from split: %s",
                kept_metadata,
                split,
            )

        # Cached number of shards.
        self._number_of_shards: int = len(shard_list)

        # First balance by file type, and each file type balance by source.
        self._shard_info_iter = _split_balancing(
            shard_list=shard_list,
            balance_by=balance_by,
            repeat=repeat,
            shuffle=shuffle,
        )

    def number_of_shards(self) -> int:
        """Return the number of distinct shards that are iterated. When
        repeated this method still returns a finite answer.
        """
        return self._number_of_shards

    def __iter__(self) -> Iterator[ShardInfo]:
        """Return the shard information iterator (reentrant).
        """
        return self

    def __next__(self) -> ShardInfo:
        """Get the next item.
        """
        return next(self._shard_info_iter)
