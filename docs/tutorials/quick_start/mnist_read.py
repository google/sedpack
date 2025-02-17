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
"""Read MNIST data and feed it to a neural network. For a tutorial with
explanations see: https://google.github.io/sedpack/tutorials/mnist

Example use:
    python mnist_save.py -d "~/Datasets/my_new_dataset/"
    python mnist_read.py -d "~/Datasets/my_new_dataset/"
"""
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sedpack.io import Dataset
from sedpack.io.types import ExampleT, TFModelT


def get_model() -> TFModelT:
    """Return a CNN model.
    """
    input_shape = (28, 28)
    num_classes = 10

    input_data = keras.Input(shape=input_shape, name="input")

    x = input_data
    x = layers.Reshape((*input_shape, 1))(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation="softmax", name="digit")(x)

    model = keras.Model(inputs=input_data, outputs=x)

    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


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

    # Load train and test and train
    model = get_model()

    dataset = Dataset(args.dataset_directory)  # Load the dataset

    # ExampleT: TypeAlias of dict[str, sedpack.io.types.AttributeValueT]
    def process_record(rec: ExampleT) -> Tuple[Any, Any]:
        output = rec["digit"]
        output = tf.one_hot(output, 10)
        return rec["input"], output

    # Load train and validation splits of the dataset
    batch_size = 128
    train_data = dataset.as_tfdataset(
        "train",
        batch_size=batch_size,
        process_record=process_record,
    )
    validation_data = dataset.as_tfdataset(
        "test",  # validation split
        batch_size=batch_size,
        process_record=process_record,
    )

    steps_per_epoch = 100
    epochs = 10
    _ = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=steps_per_epoch // 10,
    )

    # Evaluate the model on holdout.
    holdout_data = dataset.as_tfdataset(
        "holdout",
        batch_size=batch_size,
        process_record=process_record,
        repeat=False,  # Single iteration over the dataset.
    )
    score = model.evaluate(
        holdout_data,
        verbose=0,
    )
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {100 * score[1]:.2f}%")

    evaluated: int = 0
    ascii_shades = " .-/X0#"
    for example in dataset.as_numpy_iterator(split="holdout",
                                             process_record=process_record):
        # Stop after a few evaluations.
        evaluated += 1
        if evaluated >= args.ascii_evaluations:
            break

        # Pass just the input (the handwritten digit image) to the model and get
        # the predicted class as the class with highest probability.
        image = example[0]
        # Note that the model still expects a batch, here we pass a batch of one
        # image.
        predicted_class: int = np.argmax(model(np.expand_dims(image, axis=0)))
        correct_class: int = np.argmax(example[1])
        print("")
        print(f"Predicted: {predicted_class} (should be {correct_class}) for")
        # Turn into ASCII art
        for row in image:
            print("".join(
                ascii_shades[int(pixel * len(ascii_shades))] for pixel in row))


if __name__ == "__main__":
    main()
