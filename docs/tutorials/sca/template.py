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
"""Template attacks code. Based on
http://wiki.newae.com/Tutorial_B8_Profiling_Attacks_(Manual_Template_Attack)
http://wiki.newae.com/Template_Attacks#Points_of_Interest

Chari, Suresh, Josyula R. Rao, and Pankaj Rohatgi. "Template attacks."
International workshop on cryptographic hardware and embedded systems.
Berlin, Heidelberg: Springer Berlin Heidelberg, 2002.
"""

# type: ignore
import argparse
from pathlib import Path
import math
import time
from typing import NamedTuple

from jax.typing import ArrayLike
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np

from sedpack.io import Dataset
from sedpack.io.types import SplitT

# Cut the trace based on our SNR (SBOX in byte 0)
cut_middle: int = 514
radius: int = 5
cut_begin: int = max(cut_middle - radius, 0)
cut_end: int = cut_middle + radius


class OnlineTemplate(NamedTuple):
    """A pytree representing state of online template computation for a single
    byte index and all leakage values.  Based on the algorithm for online
    covariance computation:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Attributes:

      n (ArrayLike): Number of seen traces for each possible leakage value,
      shape (different_leakage_values,).

      mean (ArrayLike): Running mean for each possible leakage value, shape
      (different_leakage_values, trace_len,).

      c (ArrayLike): Running covariance matrix for each possible leakage value,
      shape (different_leakage_values, trace_len, trace_len).
    """
    n: ArrayLike
    mean: ArrayLike
    c: ArrayLike


class UpdateData(NamedTuple):
    """A pytree representing the current update.

    Attributes:

      leakage_value (jnp.int32): The leakage value. Assumed to be in
      range(different_leakage_values), see OnlineTemplate.

      trace (ArrayLike): The trace for this example, shape (trace_len,).
    """
    leakage_value: jnp.int32
    trace: ArrayLike


@jax.jit
def _update_state(
    state: OnlineTemplate,
    update_data: UpdateData,
) -> tuple[OnlineTemplate, jnp.int32]:
    """Update the online template `update_data.leakage_value` using a single
    trace.

    Args:

      state (OnlineTemplate): The current template state.

      update_data (UpdateData): The trace and leakage value determining which
      template should be updated.

    Returns: The new state of online templates computation and the number of
    traces used to create the template corresponding to the leakage value.
    """
    assert update_data.leakage_value.ndim == 0
    assert update_data.trace.ndim == 1

    prev_n = state.n[update_data.leakage_value]
    prev_mean = state.mean[update_data.leakage_value]
    prev_c = state.c[update_data.leakage_value]

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Welford's online algorithm
    n = prev_n + 1
    delta = update_data.trace - prev_mean
    mean = prev_mean + (delta / n)
    delta2 = update_data.trace - mean
    c = prev_c + jnp.einsum("i,j->ij", delta, delta2)

    return (
        OnlineTemplate(  # Carry for jax.lax.scan
            n=state.n.at[update_data.leakage_value].set(n),
            mean=state.mean.at[update_data.leakage_value].set(mean),
            c=state.c.at[update_data.leakage_value].set(c),
        ),
        n,  # Output for jax.lax.scan
    )


@jax.jit
def _log_pdf(
    trace: ArrayLike,
    det_c: ArrayLike,
    inv_c: ArrayLike,
    mean: ArrayLike,
) -> jnp.float64:
    """Compute the logarithm of probability density function for the given
    trace and single leakage value.
    """
    assert trace.ndim == 1
    assert det_c.ndim == 0
    assert inv_c.ndim == 2
    assert mean.ndim == 1

    centralized_trace = trace - mean
    return -0.5 * (jnp.log(det_c) + jnp.einsum(
        "i,ij,j",
        centralized_trace,
        inv_c,
        centralized_trace,
    ) + inv_c.shape[0] * jnp.log(2 * math.pi))


def profile_templates(dataset: Dataset, byte_index: int) -> OnlineTemplate:
    """Compute templates.
    """
    # Cut the trace based on our SNR (SBOX in byte 0)
    trace_len: int = cut_end - cut_begin
    batch_size: int = 1_024

    split: SplitT = "train"

    different_leakage_values: int = 9
    aggregate: OnlineTemplate = OnlineTemplate(
        n=jnp.zeros(different_leakage_values, dtype=jnp.int32),
        mean=jnp.zeros((different_leakage_values, trace_len)),
        c=jnp.zeros((different_leakage_values, trace_len, trace_len)),
    )

    for example in tqdm(
            dataset.as_numpy_iterator_rust_batched(
                split=split,
                repeat=False,
                shuffle=0,
                batch_size=batch_size,
            ),
            desc=f"Profiling templates - byte index {byte_index} on {split}",
            total=dataset.dataset_info.splits[split].number_of_examples //
            batch_size,
    ):
        aggregate, _ = jax.lax.scan(
            _update_state,
            aggregate,
            UpdateData(
                trace=example["trace1"][:, cut_begin:cut_end],
                leakage_value=jnp.bitwise_count(
                    example["plaintext"][:, byte_index] ^
                    example["key"][:, byte_index]),
            ),
        )

    return aggregate


def attack(dataset: Dataset,
           templates: OnlineTemplate,
           byte_index: int,
           different_target_secrets: int = 256,
           ddof: int = 0) -> None:
    covariance_matrix = templates.c / (templates.n - ddof).reshape(-1, 1, 1)
    det_c = jnp.linalg.det(covariance_matrix)
    inv_c = jnp.linalg.inv(covariance_matrix)
    mean = templates.mean

    probabilities = np.zeros(
        different_target_secrets,
        dtype=np.float64,
    )

    vmap_log_pdf = jax.jit(
        jax.vmap(
            _log_pdf,
            in_axes=(
                None,  # trace
                0,  # det_c
                0,  # inv_c
                0,  # mean
            ),
            out_axes=0,
        ))

    split = "holdout"
    use_examples: int = 10  # at most 256
    different_leakage_values: int = templates.n.shape[-1]
    real_key = None
    for i, example in enumerate(
            dataset.as_numpy_iterator_rust(
                split=split,
                repeat=False,
                shuffle=0,
            ),):
        if i >= use_examples:
            break

        if real_key is None:
            real_key = example["key"]
        else:
            assert all(real_key == example["key"])

        local_cache = vmap_log_pdf(
            example["trace1"][cut_begin:cut_end],  # trace
            det_c,  # det_c
            inv_c,  # inv_c
            mean,  # mean
        )
        assert local_cache.shape == (different_leakage_values,)

        for guess in range(different_target_secrets):
            leakage = jnp.bitwise_count(example["plaintext"][byte_index] ^
                                        guess)
            probabilities[guess] += local_cache[leakage]
        current_rank = int(
            sum(probabilities[real_key[byte_index]] <= probabilities))
        print(f"{i + 1} traces rank: {current_rank}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Do a template attack")
    parser.add_argument(
        "--dataset_path",
        "-d",
        help="Where to load the dataset",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    byte_index: int = 0
    # Load the dataset
    dataset = Dataset(args.dataset_path)

    template_begin = time.time()
    result = profile_templates(
        dataset=dataset,
        byte_index=byte_index,
    )
    template_runtime = time.time() - template_begin
    print(f"It took {template_runtime:.2f}s to profile the templates")

    attack(
        dataset=dataset,
        templates=result,
        byte_index=byte_index,
        different_target_secrets=256,
        ddof=0,
    )


if __name__ == "__main__":
    main()
