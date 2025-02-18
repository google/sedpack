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
"""Read MNIST data and feed it to a neural network. For a tutorial with
explanations see: https://google.github.io/sedpack/tutorials/mnist

Inspired by https://flax.readthedocs.io/en/latest/mnist_tutorial.html

Example use:
    python mnist_save.py -d "~/Datasets/my_new_dataset/"
    python mnist_read_jax.py -d "~/Datasets/my_new_dataset/"
"""
import argparse
from functools import partial
from typing import Any, Dict, Tuple

from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from sedpack.io import Dataset
from sedpack.io.types import ExampleT, TFModelT


def jaxify(d):
    """Turn the NumPy arrays into JAX arrays and reshape the input to have a
    channel.
    """
    batch_size: int = d["input"].shape[0]
    return {
        "input": jnp.array(d["input"]).reshape(batch_size, 28, 28, 1),
        "digit": jnp.array(d["digit"], jnp.int32),
    }


class CNN(nnx.Module):
    """FLAX CNN model.
    """

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool,
                                window_shape=(2, 2),
                                strides=(2, 2))
        self.linear1 = nnx.Linear(3_136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def loss_fn(model: CNN, batch):
    logits = model(batch["input"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["digit"]).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric,
               batch):
    """Train for a single step.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["digit"])
    optimizer.update(grads)


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["digit"])


def main() -> None:
    """Train a neural network on the MNIST dataset saved in the sedpack
    format.
    """
    parser = argparse.ArgumentParser(
        description=
        "Read MNIST in dataset-lib format and train a small neural network.")
    parser.add_argument("--dataset_directory",
                        "-d",
                        help="Where to load the dataset",
                        required=True)
    parser.add_argument("--ascii_evaluations",
                        "-e",
                        help="How many images to print and evaluate",
                        type=int,
                        default=10)
    args = parser.parse_args()

    model = CNN(rngs=nnx.Rngs(0))
    nnx.display(model)

    learning_rate = 0.005
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    nnx.display(optimizer)

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    dataset = Dataset(args.dataset_directory)  # Load the dataset
    batch_size = 32
    train_data = dataset.as_tfdataset(
        "train",
        batch_size=batch_size,
        shuffle=1_000,
    )
    validation_data = dataset.as_tfdataset(
        "test",  # validation split
        batch_size=batch_size,
        shuffle=1_000,
        repeat=False,
    )
    train_steps = 1_200
    eval_every = 200

    for step, batch in enumerate(tqdm(train_data)):
        if step > train_steps:
            break

        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics
        batch = jaxify(batch)
        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step
                         == train_steps - 1):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items(
            ):  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(
                    value)  # Record the metrics.
                print(f"{metric} = {value}", end=" ")
            metrics.reset()  # Reset the metrics for the test set.
            print()

            # Compute the metrics on the test set after each training epoch.
            for test_batch in validation_data.as_numpy_iterator():
                test_batch = jaxify(test_batch)
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
                print(f"test {metric} = {value}", end=" ")
            metrics.reset()  # Reset the metrics for the next training epoch.
            print()


if __name__ == "__main__":
    main()
