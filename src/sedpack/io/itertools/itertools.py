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

import random
from typing import AsyncIterable, Iterable, TypeVar

import asyncstdlib
import numpy as np


def initial_random_state(seed: int | None = None) -> np.uint32:
    """Get a random seed. Defaults to using random.randint.

    Args:

      seed (int | None): Define the seed or use a random value.
    """
    if seed is None:
        seed = random.randint(0, 2**30)

    # Make sure what we return is really np.uint32 and not Python int.
    return np.array([seed], np.uint32)[0]  # type: ignore[no-any-return]


def next_random_state(r: np.uint32) -> np.uint32:
    """Linear congruential generator get next state. See the book "Numerical
    Recipes in C" 2nd edition, Press et al., isbn 0521431085, p. 284

    Args:

      r (np.uint32): Previous state.

    Returns the next state.
    """
    return r * 1664525 + 1013904223  # type: ignore[no-any-return]


T = TypeVar("T")


async def shuffle_buffer_async(iterable: AsyncIterable[T],
                               buffer_size: int) -> AsyncIterable[T]:
    """Shuffle buffer to randomize iterating over a potentially very long
    sequence.

    Args:

      iterable (AsyncIterable[T]): Potentially a very long sequence.

      buffer_size (int): Maximum number of elements we keep in memory at the
      same time.

    TODO: add random generator as a parameter if needed.
    """
    # Otherwise the first elements of a list would be iterated multiple times.
    iterable = aiter(iterable)

    # Fill the buffer.
    buffer: list[T] = []
    async for _, item in asyncstdlib.zip(range(buffer_size), iterable):
        buffer.append(item)

    # Iterate and keep filling the buffer.
    r = initial_random_state()
    while True:
        try:
            new_element = await anext(iterable)
        except StopAsyncIteration:
            break

        # Which element to yield and replace.
        i = r % len(buffer)
        yield buffer[i]
        buffer[i] = new_element

        r = next_random_state(r)

    # Finish what is left.
    random.shuffle(buffer)
    for element in buffer:
        yield element


def shuffle_buffer(iterable: Iterable[T], buffer_size: int) -> Iterable[T]:
    """Shuffle buffer to randomize iterating over a potentially very long
    sequence.

    Args:

      iterable (Iterable[T]): Potentially a very long sequence.

      buffer_size (int): Maximum number of elements we keep in memory at the
      same time.

    TODO: add random generator as a parameter if needed.
    """
    # Otherwise the first elements of a list would be iterated multiple times.
    iterable = iter(iterable)

    # Fill the buffer.
    buffer: list[T] = []
    for _, item in zip(range(buffer_size), iterable):
        buffer.append(item)

    # Iterate and keep filling the buffer.
    r = initial_random_state()
    while True:
        try:
            new_element = next(iterable)
        except StopIteration:
            break

        # Which element to yield and replace.
        i = r % len(buffer)
        yield buffer[i]
        buffer[i] = new_element

        r = next_random_state(r)

    # Finish what is left.
    random.shuffle(buffer)
    yield from buffer


def round_robin(iterables: Iterable[Iterable[T]],
                buffer_size: int = 5) -> Iterable[T]:
    """Visit input iterables in a cycle until each is exhausted interleaving
    them in the process. Automatically shuffles.

    # roundrobin('ABC', 'D', 'EF') -> A D E B F C (in random order)
    # Algorithm credited to George Sakkis
    # source https://docs.python.org/3/library/itertools.html

    Args:

      iterables (Iterable[Iterable[T]]): A potentially infinite iterable of
      iterables.

      buffer_size (int): Size of the internal buffer.
    """
    # Otherwise the first elements of a list would be iterated multiple times.
    iterables = iter(iterables)
    buffer = []

    # Fill in the buffer.
    for _, i in zip(range(buffer_size), iterables):
        buffer.append(iter(i))

    r = initial_random_state()

    # Use the buffer and refill as needed.
    while buffer:
        pos = r % len(buffer)
        r = next_random_state(r)

        try:
            yield next(buffer[pos])
        except StopIteration:
            try:
                # Try to refill the buffer.
                buffer[pos] = iter(next(iterables))
            except StopIteration:
                # Keep using the buffer until finished.
                buffer[pos] = buffer[-1]
                del buffer[-1]


async def round_robin_async(iterables: AsyncIterable[AsyncIterable[T]],
                            buffer_size: int = 5) -> AsyncIterable[T]:
    """Visit input iterables in a cycle until each is exhausted interleaving
    them in the process. Automatically shuffles.

    # roundrobin('ABC', 'D', 'EF') -> A D E B F C (in random order)
    # Algorithm credited to George Sakkis
    # source https://docs.python.org/3/library/itertools.html

    Args:

      iterables (AsyncIterable[AsyncIterable[T]]): A potentially infinite
      iterable of iterables.

      buffer_size (int): Size of the internal buffer.
    """
    iterables = aiter(iterables)
    buffer = []

    # Fill in the buffer.
    for _ in range(buffer_size):
        try:
            buffer.append(aiter(await anext(iterables)))
        except StopAsyncIteration:
            break

    r = initial_random_state()

    # Use the buffer and refill as needed.
    while buffer:
        pos = r % len(buffer)
        r = next_random_state(r)

        try:
            yield await anext(buffer[pos])
        except StopAsyncIteration:
            try:
                # Try to refill the buffer.
                buffer[pos] = aiter(await anext(iterables))
            except StopAsyncIteration:
                # Keep using the buffer until finished.
                buffer[pos] = buffer[-1]
                del buffer[-1]
