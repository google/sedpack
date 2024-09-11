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
"""Iteration helpers.
"""

import itertools
import os
import queue
import time
import threading
from types import TracebackType
from typing import Any, Callable, Iterable, Type, TypeVar
from typing_extensions import Self

U = TypeVar("U")
V = TypeVar("V")


class StopSentinel:  # pylint: disable=too-few-public-methods
    """Nothing more is coming.
    """


class LazyPool:
    """Lazy version of `concurrent.futures.ThreadPoolExecutor.map`. Allows to
    iterate content of shards without reading all of them into memory if they
    are iterated slower than read.

    Example use:
    ```
      f = lambda x: 2 * x

      with LazyPool() as pool:
        for twice_i in pool.imap_unordered(f, range(10)):
          print(twice_i)
      ```
      """

    def __init__(self, threads: int | None = os.cpu_count()) -> None:
        """Prepare for `imap_unordered` call.

        Args:

          threads (int | None): How many threads should be used concurrently.
        """
        # How many threads to use.
        self._threads: int = max(1, threads or 1)

        # Main communication queues.
        self._to_process: queue.Queue[Any] | None = None
        self._results: queue.Queue[Any] | None = None

        # Nothing is active yet.
        self._active_threads: int = 0

    def __enter__(self) -> Self:
        # Make sure that we do not start another iteration before finishing the
        # old one.
        assert self._active_threads <= 0
        assert self._to_process is None
        assert self._results is None
        return self

    def finish_and_reset(self) -> None:
        """Reset state for a possible next call of `imap_unordered`. Finish all
        running threads.
        """
        self._active_threads = 0
        self._results = None
        # Stop threads if needed.
        if self._to_process is None:
            return
        # Send stop to all threads.
        for _ in range(self._threads):
            self._to_process.put(StopSentinel())
        self._to_process = None

    def __exit__(self, exc_type: Type[BaseException] | None,
                 exc: BaseException | None,
                 exc_tb: TracebackType | None) -> bool:
        self.finish_and_reset()
        if exc:
            raise exc
        return True

    def imap_unordered(self, func: Callable[[U], V],
                       iterable: Iterable[U]) -> Iterable[V]:
        """Lazy map of `func` on the `iterable`.

        Args:

          func (Callable[[U], V]): The function to be applied.

          iterable (Iterable[U]): The iterable with inputs.

        Returns: `Iterable[V]` iterable of results. Which might be out of
        order.
        """
        # Make sure that we do not start another iteration before finishing the
        # old one.
        assert self._active_threads <= 0
        assert self._to_process is None
        assert self._results is None
        self._to_process = queue.Queue[U | StopSentinel]()
        self._results = queue.Queue[V | StopSentinel]()

        iterator_with_stops: Iterable[U | StopSentinel] = itertools.chain(
            iterable, itertools.cycle([StopSentinel()]))
        del iterable  # We no longer use `iterable`.
        # Make sure that we do not iterate the initial inputs twice.
        iterator_with_stops = iter(iterator_with_stops)

        # Start process communication threads.
        collectors: list[Collector] = [
            Collector(
                func=func,
                to_process=self._to_process,
                results=self._results,
            ) for _ in range(self._threads)
        ]
        for collector in collectors:
            collector.start()
        # All threads started.
        self._active_threads = self._threads

        # Assign some work to the threads.
        for i, element in enumerate(iterator_with_stops):
            self._to_process.put(element)
            # One is read and another one should start processing
            # immediately.
            if i > 2 * self._threads:
                break

        # Get and yield one and put another to be processed.
        while self._active_threads > 0:
            next_result: V | StopSentinel = self._results.get()
            if isinstance(next_result, StopSentinel):
                self._active_threads -= 1
                continue

            # New element to be processed. After the potentially finite
            # `iterator` we append an infinite number of `StopSentinel`s so the
            # `next` does not raise `StopIteration`.
            try:
                # Avoid pydantic stop-iteration-return warning.
                self._to_process.put(next(iterator_with_stops))
            except StopIteration as exc:
                raise ValueError("StopSentinel missing.") from exc
            yield next_result

        # Allow multiple calls.
        self.finish_and_reset()


class Collector(threading.Thread):
    """Move elements from the `to_process` queue and put them into the
    `results` queue.
    """

    def __init__(self, func: Callable[[U], V],
                 to_process: queue.Queue[U | StopSentinel],
                 results: queue.Queue[V | StopSentinel], *args: Any,
                 **kwargs: Any) -> None:
        """Take take elements to process from `to_process` queue and put the
        results into `results` (both queues are shared with other threads).
        When it sees a `StopSentinel` instance it puts it immediately into
        `results` and stops (consumes at most one `StopSentinel` and returns
        it).

        Args:

          func (Callable[[U], V]): The function to be applied.

          to_process (queue.Queue[U | StopSentinel]): Incoming items.

          results (queue.Queue[V | StopSentinel]): Return back the results.

          *args, **kwargs: Passed to `Thread` constructor.
        """
        super().__init__(*args, **kwargs)

        # Main communication queues from LazyPool.
        self._to_process: queue.Queue[U | StopSentinel] = to_process
        self._results: queue.Queue[V | StopSentinel] = results
        self.func: Callable[[U], V] = func

    def run(self) -> None:
        """Run this thread.
        """
        while True:
            try:
                element = self._to_process.get()
            except queue.Empty:
                time.sleep(0.0)  # Give up GIL.
                continue

            if isinstance(element, StopSentinel):
                # Nothing more to process. Invariant is that each `Collector`
                # consumes at most single `StopSentinel` from the queue
                # `self._to_process`.  Thus to stop all `Collector`s and their
                # threads it is enough to feed as many `StopSentinel` values as
                # there are `Collector`s.
                self._results.put(element)
                return

            # Can be blocking, but should be short.
            self._results.put(self.func(element))
            time.sleep(0.0)  # Give up GIL.
