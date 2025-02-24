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
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from scaaml.stats.snr import SNRSinglePass
from scaaml.stats.attack_points.aes_128.attack_points import (
    LeakageModelAES128,
    SubBytesIn,
)
from sedpack.io import Dataset
from sedpack.io.types import SplitT

import jax
import jax.numpy as jnp


@jax.jit  # type: ignore[misc]
def jax_update(existing_aggregate: jax.Array,
               new_trace: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """For a new value new_trace, compute the new count, new mean, the new
    squared_deltas.  mean accumulates the mean of the entire dataset
    squared_deltas aggregates the squared distance from the mean count
    aggregates the number of samples seen so far
    """
    (count, mean, squared_deltas) = existing_aggregate
    count += 1
    delta = new_trace - mean
    mean += delta / count
    updated_delta = new_trace - mean
    squared_deltas += delta * updated_delta
    return (count, mean, squared_deltas)


def jax_get_initial_aggregate(
        trace_len: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return an initial aggregate.
    """
    dtype = jnp.float32
    count = jnp.array(0, dtype=dtype)
    mean = jnp.zeros(trace_len, dtype=dtype)
    squared_deltas = jnp.zeros(trace_len, dtype=dtype)
    return (count, mean, squared_deltas)


# Retrieve the mean, variance and sample variance from an aggregate
def jax_finalize(
    existing_aggregate: tuple[jax.Array, jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Retrieve the mean, variance, and sample variance from an aggregate.
    """
    (count, mean, squared_deltas) = existing_aggregate
    assert count >= 2
    return (
        mean,
        squared_deltas / count,
    )


def snr_jax(dataset_path: Path, ap_name: str) -> npt.NDArray[np.float32]:
    """Compute SNR using NumPy.
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    # We know that trace1 is the first.
    trace_len: int = dataset.dataset_structure.saved_data_description[0].shape[
        0]

    leakage_to_aggregate = {
        i: jax_get_initial_aggregate(trace_len=trace_len) for i in range(9)
    }

    split: SplitT = "test"

    for example in tqdm(
            dataset.as_numpy_iterator(
                split=split,
                repeat=False,
                shuffle=0,
            ),
            desc=f"[JAX] Computing SNR over {split}",
            total=dataset.dataset_info.splits[split].number_of_examples,
    ):
        current_leakage = int(
            example[ap_name][0],  # type: ignore[index]
        ).bit_count()
        leakage_to_aggregate[current_leakage] = jax_update(
            leakage_to_aggregate[current_leakage],
            example["trace1"],
        )

    results = {
        leakage: jax_finalize(aggregate)
        for leakage, aggregate in leakage_to_aggregate.items()
    }

    # Find out which class is the most common.
    most_common_leakage = 0
    most_common_count = 0
    for leakage, (count, _, _) in leakage_to_aggregate.items():
        if count >= most_common_count:
            most_common_leakage = leakage
            most_common_count = count

    signals = np.array([mean for mean, _ in results.values()])

    return np.array(
        20 * np.log(np.var(signals, axis=0) / results[most_common_leakage][1]),
        dtype=np.float32,
    )


def snr_np(dataset_path: Path) -> npt.NDArray[np.float64]:
    """Compute SNR using NumPy.
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    snr = SNRSinglePass(
        leakage_model=LeakageModelAES128(
            byte_index=0,
            attack_point=SubBytesIn(),
            use_hamming_weight=True,
        ),
        db=True,
    )

    split: SplitT = "test"

    for example in tqdm(
            dataset.as_numpy_iterator(
                split=split,
                repeat=False,
                shuffle=0,
            ),
            desc=f"[NP] Computing SNR over {split}",
            total=dataset.dataset_info.splits[split].number_of_examples,
    ):
        snr.update(example)

    plt.plot(snr.result)
    plt.savefig("snr.png")
    return np.array(snr.result, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_path",
                        "-d",
                        help="Where to load the dataset",
                        type=Path,
                        required=True)
    args = parser.parse_args()

    result_jax = snr_jax(dataset_path=args.dataset_path, ap_name="sub_bytes_in")
    result_np_slow = snr_np(dataset_path=args.dataset_path)

    print(f"{np.max(np.abs(result_np_slow - result_jax)) = }")


if __name__ == "__main__":
    main()
