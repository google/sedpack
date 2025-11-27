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

Compute CPA https://wiki.newae.com/Correlation_Power_Analysis.

Example use:
    python cpa.py --dataset_path "~/datasets/tiny_aes_sedpack/"
"""
import argparse
from functools import partial
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from sedpack.io import Dataset
from sedpack.io.types import SplitT

from scaaml.stats.cpa import CPA
from scaaml.stats.attack_points.aes_128.full_aes import encrypt
from scaaml.stats.attack_points.aes_128.attack_points import SubBytesIn, LeakageModelAES128

# Cut the trace
trace_len: int = 10_000
batch_size: int = 64
split: SplitT = "train"


class UpdateData(NamedTuple):
    """A pytree representing the current update.

    Attributes:

      trace (ArrayLike): The trace for this example, shape (trace_len,).

      leakage_value (ArrayLike): The leakage value given the guess. Assumed to
      be in range(different_leakage_values). The shape is
      (different_target_secrets,).
    """
    trace: ArrayLike
    hypothesis: ArrayLike


def get_initial_aggregate(
    trace_len: int,
    different_target_secrets: int = 256,
) -> dict[str, ArrayLike]:
    """Return an initial aggregate for a single byte index.

    Args:

      trace_len (int): The length of a single trace (or number of points of
      interest if you cut the trace).

      different_target_secrets (int): How many values can the secret have. Most
      likely we are trying to infer a byte value (even when the leakage model
      is Hamming weight).

    Returns: A pytree representing state of online SNR computation for a single
    byte index.

    Keys and values:

      d (ArrayLike): The number of seen examples, shape (1,).

      sum_h_t (ArrayLike): The running sum outer products of hypothesis values
      and trace, shape (different_target_secrets, trace_len).

      sum_h (ArrayLike): The running sum of all hypothesis, shape
      (different_target_secrets,).

      sum_hh (ArrayLike): The running sum of squares of all hypothesis values,
      shape (different_target_secrets,).

      sum_t (ArrayLike): The running sum of all traces, shape (trace_len,).

      sum_tt (ArrayLike): The running sum of squares all traces, shape
      (trace_len,).
    """
    dtype = jnp.float32
    return {
        "d":
            jnp.zeros(1, dtype=jnp.int64),
        "sum_h_t":
            jnp.zeros((different_target_secrets, trace_len), dtype=dtype),
        "sum_h":
            jnp.zeros(different_target_secrets, dtype=dtype),
        "sum_hh":
            jnp.zeros(different_target_secrets, dtype=dtype),
        "sum_t":
            jnp.zeros(trace_len, dtype=dtype),
        "sum_tt":
            jnp.zeros(trace_len, dtype=dtype),
    }


def get_initial_aggregate_multibyte(
    trace_len: int,
    different_target_secrets: int = 256,
    num_byte_indexes: int = 16,
) -> dict[str, ArrayLike]:
    """Return an initial aggregate for a all byte indices at once.

    Args:

      trace_len (int): The length of a single trace (or number of points of
      interest if you cut the trace).

      different_target_secrets (int): How many values can the secret have. Most
      likely we are trying to infer a byte value (even when the leakage model
      is Hamming weight).

      num_byte_indexes (int): Defaults to 16 but could be more, e.g., in case
      of AES256.

    Returns: A pytree representing state of online SNR computation for a single
    byte index.

    Keys and values:

      d (ArrayLike): The number of seen examples, shape (1,).

      sum_h_t (ArrayLike): The running sum outer products of hypothesis values
      and trace, shape (num_byte_indexes, different_target_secrets, trace_len).

      sum_h (ArrayLike): The running sum of all hypothesis, shape
      (num_byte_indexes, different_target_secrets,).

      sum_hh (ArrayLike): The running sum of squares of all hypothesis values,
      shape (num_byte_indexes, different_target_secrets,).

      sum_t (ArrayLike): The running sum of all traces, shape (trace_len,).

      sum_tt (ArrayLike): The running sum of squares all traces, shape
      (trace_len,).
    """
    dtype = jnp.float32
    return {
        "d":
            jnp.zeros(1, dtype=jnp.int64),
        "sum_h_t":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets, trace_len),
                dtype=dtype,
            ),
        "sum_h":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets),
                dtype=dtype,
            ),
        "sum_hh":
            jnp.zeros(
                (num_byte_indexes, different_target_secrets),
                dtype=dtype,
            ),
        "sum_t":
            jnp.zeros(
                trace_len,
                dtype=dtype,
            ),
        "sum_tt":
            jnp.zeros(
                trace_len,
                dtype=dtype,
            ),
    }


@jax.jit
def r_update(
    state: dict[str, ArrayLike],
    data: UpdateData,
) -> (dict[str, ArrayLike], jnp.int32):
    """Update the CPA aggregate state.
    """
    # Check the dimensions if debugging. This will work even across vmaps, jit,
    # scan, etc.
    assert data.trace.shape == state["sum_t"].shape
    assert data.hypothesis.shape == state["sum_h"].shape

    # D (so far)
    d = state["d"] + 1
    # i indexes the hypothesis possible values
    # j indexes the time dimension

    # \sum_{d=1}^{D} h_{d,i} t_{d,j}
    sum_h_t = state["sum_h_t"] + jnp.einsum("i,j->ij", data.hypothesis,
                                            data.trace)

    # \sum_{d=1}^{D} h_{d, i}
    sum_h = state["sum_h"] + data.hypothesis

    # \sum_{d=1}^{D} t_{d, j}
    sum_t = state["sum_t"] + data.trace

    # \sum_{d=1}^{D} h_{d, i}^2
    sum_hh = state["sum_hh"] + data.hypothesis**2

    # \sum_{d=1}^{D} t_{d, j}^2
    sum_tt = state["sum_tt"] + data.trace**2

    return (
        {
            "d": d,
            "sum_h_t": sum_h_t,
            "sum_h": sum_h,
            "sum_hh": sum_hh,
            "sum_t": sum_t,
            "sum_tt": sum_tt,
        },
        d,
    )


@partial(jax.jit, static_argnames=["return_absolute_value"])
def r_guess_with_time(
    state: dict[str, ArrayLike],
    return_absolute_value: bool,
) -> ArrayLike:
    nom = (state["d"] * state["sum_h_t"]) - jnp.einsum(
        "i,j->ij", state["sum_h"], state["sum_t"])

    # denominator squared
    den_a = (state["sum_h"]**2) - (state["d"] * state["sum_hh"])  # i
    den_b = (state["sum_t"]**2) - (state["d"] * state["sum_tt"])  # j

    r = nom / jnp.sqrt(jnp.einsum("i,j->ij", den_a, den_b))

    if return_absolute_value:
        return jnp.abs(r)
    return r


@partial(jax.jit, static_argnames=["return_absolute_value"])
def r_guess_notime(
    state: dict[str, ArrayLike],
    return_absolute_value: bool,
) -> ArrayLike:
    # Forget time axis.
    return jnp.max(
        r_guess_with_time(
            state,
            return_absolute_value=return_absolute_value,
        ),
        axis=1,
    )


def print_ranks(full_guess: ArrayLike, real_key: ArrayLike) -> None:
    """Prints how many target values had higher probability than the real
    secret.

    Args:

      full_guess (ArrayLike): The probabilities of shape (16, 256) as returned
      by `r_guess_notime`.

      real_key (ArrayLike): The real secret value of shape (16,) of np.uint8.
    """
    print("Ranks of the real guessed values, the lower the better, 1 is top "
          "prediction:")
    print([
        int(np.sum(full_guess[i, :] >= full_guess[i, real_key[i]]))
        for i in range(16)
    ])


def cpa_single_byte(dataset_path: Path) -> npt.NDArray[np.float32]:
    """Compute SNR using NumPy.
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    aggregate = get_initial_aggregate(
        trace_len=trace_len,
        different_target_secrets=256,  # Predicting a byte value.
    )

    byte_index: int = 7

    for example in tqdm(
            dataset.as_numpy_iterator_rust(
                split=split,
                repeat=False,
                shuffle=0,
            ),
            desc=f"[JAX] Computing single byte index CPA over {split}",
            total=dataset.dataset_info.splits[split].number_of_examples,
    ):
        # Simulated lekage guesses.
        simulated_plaintext = example["plaintext"][byte_index] ^ example["key"][
            byte_index]
        hypothesis = jnp.bitwise_count(simulated_plaintext ^
                                       np.arange(256, dtype=np.uint8))

        aggregate, _ = r_update(
            aggregate,
            UpdateData(
                trace=example["trace1"][:trace_len],
                hypothesis=hypothesis,
            ),
        )

    #print(f"{r_guess_notime(aggregate, return_absolute_value=True) = }")
    guessed = int(np.argmax(r_guess_notime(aggregate, return_absolute_value=True,)))
    print(f"{guessed = } (correct answer is: 0 -- simulated key)")
    return guessed


def cpa_single_experiment(dataset_path: Path) -> npt.NDArray[np.float32]:
    """
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    # AES128
    num_byte_indexes: int = 16

    aggregate = get_initial_aggregate_multibyte(
        trace_len=trace_len,
        different_target_secrets=256,  # Predicting a byte value.
        num_byte_indexes=num_byte_indexes,
    )

    aggregate_vmap = {
        "d": None,
        "sum_h_t": 0,
        "sum_h": 0,
        "sum_hh": 0,
        "sum_t": None,
        "sum_tt": None,
    }
    r_update_multiindex = jax.jit(
        jax.vmap(
            r_update,
            in_axes=(
                aggregate_vmap,
                UpdateData(
                    trace=None,
                    hypothesis=0,
                ),
            ),
            out_axes=(
                aggregate_vmap,
                None,
            ),
        ))

    real_key = None

    desc = f"[JAX] Computing CPA with 256 examples with fixed key in {split}"
    for i, example in enumerate(
            tqdm(
                dataset.as_numpy_iterator_rust(
                    split=split,
                    repeat=False,
                    shuffle=0,
                ),
                desc=desc,
                total=256,
            )):
        if i >= 256:
            break

        if real_key is not None:
            assert all(real_key == example["key"])
        else:
            real_key = example["key"]

        plaintext = example["plaintext"]
        hypothesis = jnp.bitwise_count(
            plaintext.reshape(num_byte_indexes, 1) ^
            np.arange(256, dtype=np.uint8).reshape(1, 256))

        aggregate, _ = r_update_multiindex(
            aggregate,
            UpdateData(
                trace=example["trace1"][:trace_len],
                hypothesis=hypothesis,
            ),
        )
    full_guess = jax.vmap(
        r_guess_notime,
        in_axes=(aggregate_vmap, None),
        out_axes=0,
    )(aggregate, True)
    guessed = np.argmax(
        full_guess,
        axis=-1,
    ).tolist()
    print_ranks(
        full_guess=full_guess,
        real_key=real_key,
    )
    return guessed


def cpa(dataset_path: Path) -> npt.NDArray[np.float32]:
    """
    """
    # Load the dataset
    dataset = Dataset(dataset_path)

    # AES128
    num_byte_indexes: int = 16

    aggregate = get_initial_aggregate_multibyte(
        trace_len=trace_len,
        different_target_secrets=256,  # Predicting a byte value.
        num_byte_indexes=num_byte_indexes,
    )

    aggregate_vmap = {
        "d": None,
        "sum_h_t": 0,
        "sum_h": 0,
        "sum_hh": 0,
        "sum_t": None,
        "sum_tt": None,
    }
    r_update_multiindex = jax.jit(
        jax.vmap(
            r_update,
            in_axes=(
                aggregate_vmap,
                UpdateData(
                    trace=None,
                    hypothesis=0,
                ),
            ),
            out_axes=(
                aggregate_vmap,
                None,
            ),
        ))

    for example in tqdm(
            dataset.as_numpy_iterator_rust_batched(
                split=split,
                repeat=False,
                shuffle=0,
                batch_size=batch_size,
            ),
            desc=f"[JAX] Computing CPA over batches of {split}",
            total=dataset.dataset_info.splits[split].number_of_examples //
            batch_size,
    ):
        # Since the dataset was not created using a constant key we need to
        # simulate. Our laekage model is the Hamming weight of S-BOX inputs
        # which had high SNR. When we would be running with constant secret key
        # our simulated_plaintext would equal the real plaintext (which we
        # assume to know). With changes of the key we could simulate all zero
        # key by setting simulated_plaintext to plaintext ^ key.
        simulated_plaintext = example["plaintext"] ^ example["key"]
        hypothesis = jnp.bitwise_count(
            simulated_plaintext.reshape(-1, num_byte_indexes, 1) ^
            jnp.arange(256, dtype=jnp.uint8).reshape(1, 1, 256))

        aggregate, _ = jax.lax.scan(
            r_update_multiindex,
            aggregate,
            UpdateData(
                trace=example["trace1"][:, :trace_len],
                hypothesis=hypothesis,
            ),
        )

    full_guess = jax.vmap(
        r_guess_notime,
        in_axes=(aggregate_vmap, None),
        out_axes=0,
    )(aggregate, True)
    guessed = np.argmax(
        full_guess,
        axis=-1,
    ).tolist()
    print_ranks(
        full_guess=full_guess,
        real_key=np.zeros(16, np.uint8),  # simulated all zeros key
    )
    return guessed


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
    result_jax = cpa_single_byte(dataset_path=args.dataset_path)
    jax_runtime = time.time() - jax_begin

    jax_begin = time.time()
    _ = cpa_single_experiment(dataset_path=args.dataset_path)
    jax_runtime = time.time() - jax_begin

    jax_begin = time.time()
    result_jax = cpa(dataset_path=args.dataset_path)
    jax_runtime = time.time() - jax_begin


if __name__ == "__main__":
    main()
