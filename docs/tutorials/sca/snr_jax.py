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
"""Side channel attack tutorial.

Compute and plot signal to noise ratio.

Be sure to install SCAAML: python -m pip install scaaml

Example use:
    python snr.py --dataset_path "~/datasets/tiny_aes_sedpack/"
"""
# type: ignore
import argparse
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from sedpack.io import Dataset
from sedpack.io.types import SplitT

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class SnrAggregate(NamedTuple):
    """A pytree representing state of online SNR computation for all possible
    leakage values together.

    Attributes:

      count (ArrayLike): Number of seen traces, shape
      (different_leakage_values,).

      mean (ArrayLike): Running mean of the trace, shape
      (different_leakage_values, trace_len).

      squared_deltas (ArrayLike): Running sum of squared distances from the
      mean, shape (different_leakage_values, trace_len).
    """
    count: ArrayLike
    mean: ArrayLike
    squared_deltas: ArrayLike


class UpdateData(NamedTuple):
    """A pytree representing the current update.

    Attributes:

      leakage_value (jnp.int32): The leakage value. Assumed to be in
      range(different_leakage_values), see SnrAggregate.

      trace (ArrayLike): The trace for this example, shape (trace_len,).
    """
    leakage_value: jnp.int32
    trace: ArrayLike


def jax_get_initial_aggregate(
    trace_len: int,
    different_leakage_values: int,
) -> SnrAggregate:
    """Return an initial aggregate for all leakage values at once.

    Args:

      trace_len (int): The length of a single trace (or number of points of
      interest if you cut the trace).

      different_leakage_values (int): It is assumed that each leakage value is
      in `range(different_leakage_values)`. Typical values are 256 or 9 in case
      of Hamming weight.
    """
    dtype = jnp.float32
    return SnrAggregate(
        count=jnp.zeros(
            (different_leakage_values,),
            dtype=dtype,
        ),
        mean=jnp.zeros(
            (different_leakage_values, trace_len),
            dtype=dtype,
        ),
        squared_deltas=jnp.zeros(
            (different_leakage_values, trace_len),
            dtype=dtype,
        ),
    )


def jax_finalize(aggregate: SnrAggregate) -> tuple[jax.Array, jax.Array]:
    """Retrieve the mean and sample variance from an aggregate.
    """
    assert jnp.all(aggregate.count >= 2)
    return (
        aggregate.mean,
        aggregate.squared_deltas / aggregate.count.reshape((-1, 1)),
    )


@jax.jit
def jax_update(
    aggregate: SnrAggregate,
    data: UpdateData,
) -> SnrAggregate:
    """For a given update of trace and leakage_value update the aggregate
    (single example, not batched). Returns the aggregate update and the total
    number of updates to be directly usable with jax.lax.scan.
    """
    count = aggregate.count.at[data.leakage_value].add(1)
    delta = data.trace - aggregate.mean[data.leakage_value]
    mean = aggregate.mean.at[data.leakage_value].add(delta /
                                                     count[data.leakage_value])
    updated_delta = data.trace - mean[data.leakage_value]
    squared_deltas = aggregate.squared_deltas.at[data.leakage_value].add(
        delta * updated_delta)
    return (
        SnrAggregate(
            count=count,
            mean=mean,
            squared_deltas=squared_deltas,
        ),
        count,  # To be usable with jax.lax.scan.
    )


@jax.jit
def jax_update_b(
    aggregate: SnrAggregate,
    leakage_values: ArrayLike,
    new_traces: ArrayLike,
) -> SnrAggregate:
    """Batched version of jax_update without returning the count.

    Args:

      aggregate (SnrAggregate): The current state.

      leakage_values (ArrayLike): The leakage values of shape (batch_size,).
      Each of those is in range(different_leakage_values).

      new_traces (ArrayLike): The batch of traces of shape (batch_size,
      trace_len).

    Returns: the final SnrAggregate as if updating batch_size times using
    jax_update (and forgetting count).
    """
    new_aggregate, _ = jax.lax.scan(
        jax_update,
        aggregate,
        UpdateData(
            leakage_value=leakage_values,
            trace=new_traces,
        ),
    )
    return new_aggregate


def snr_jax(
    dataset_path: Path,
    ap_name: str,
) -> npt.NDArray[np.float32]:
    """Compute SNR using NumPy.
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    # We know that trace1 is the first.
    trace_len: int = dataset.dataset_structure.saved_data_description[0].shape[
        0]

    leakage_aggregate = jax_get_initial_aggregate(
        trace_len=trace_len,
        different_leakage_values=9,  # Hamming weight
    )

    split: SplitT = "test"
    byte_index: int = 0

    for example in tqdm(
            dataset.as_numpy_iterator_rust(
                split=split,
                repeat=False,
                shuffle=0,
            ),
            desc=f"[JAX] Computing SNR over {split}",
            total=dataset.dataset_info.splits[split].number_of_examples,
    ):
        current_leakage = jnp.bitwise_count(example[ap_name][byte_index])
        leakage_aggregate, _ = jax_update(
            leakage_aggregate,
            UpdateData(
                leakage_value=current_leakage,
                trace=example["trace1"],
            ),
        )

    finalized = jax_finalize(leakage_aggregate)
    results = {
        int(i): (finalized[0][i], finalized[1][i])
        for i in range(finalized[0].shape[0])
    }

    # Find out which class is the most common.
    most_common_leakage = int(jnp.argmax(leakage_aggregate[0]))

    signals = np.array([mean for mean, _ in results.values()])

    return np.array(
        20 * np.log(np.var(signals, axis=0) / results[most_common_leakage][1]),
        dtype=np.float32,
    )


def snr_jax_batched(
    dataset_path: Path,
    ap_name: str,
) -> npt.NDArray[np.float32]:
    """Compute SNR using NumPy.
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    # We know that trace1 is the first.
    trace_len: int = dataset.dataset_structure.saved_data_description[0].shape[
        0]

    leakage_aggregate = jax_get_initial_aggregate(
        trace_len=trace_len,
        different_leakage_values=9,  # Hamming weight
    )

    split: SplitT = "test"
    byte_index: int = 0
    batch_size: int = 64

    for example in tqdm(
            dataset.as_numpy_iterator_rust_batched(
                split=split,
                repeat=False,
                shuffle=0,
                batch_size=batch_size,
            ),
            desc=f"[JAX] Computing SNR in batches over {split}",
            total=dataset.dataset_info.splits[split].number_of_examples //
            batch_size,
    ):
        current_leakage = jnp.bitwise_count(example[ap_name][:, byte_index])
        leakage_aggregate = jax_update_b(
            leakage_aggregate,
            current_leakage,
            example["trace1"],
        )

    finalized = jax_finalize(leakage_aggregate)
    results = {
        int(i): (finalized[0][i], finalized[1][i])
        for i in range(finalized[0].shape[0])
    }

    # Find out which class is the most common.
    most_common_leakage = int(jnp.argmax(leakage_aggregate[0]))

    signals = np.array([mean for mean, _ in results.values()])

    return np.array(
        20 * np.log(np.var(signals, axis=0) / results[most_common_leakage][1]),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_path",
        "-d",
        help="Where to load the dataset",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    jax_begin = time.time()
    result = snr_jax(
        dataset_path=args.dataset_path,
        ap_name="sub_bytes_in",
    )
    jax_runtime = time.time() - jax_begin

    jax_batched_begin = time.time()
    result_batched_computation = snr_jax_batched(
        dataset_path=args.dataset_path,
        ap_name="sub_bytes_in",
    )
    jax_batched_runtime = time.time() - jax_batched_begin

    print(f"It took {jax_runtime:.2f}s to run the non-batched version and "
          f"{jax_batched_runtime:.2f}s to run the batched version")
    print("The result differs by at most "
          f"{np.max(np.abs(result_batched_computation - result)) = }")


if __name__ == "__main__":
    main()
