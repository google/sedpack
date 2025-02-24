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
"""Side channel attack tutorial.

https://github.com/google/scaaml/tree/main/scaaml_intro

The SCAAML package is a requirement: `python3 -m pip install "scaaml>=3.0.3"`.

Example use:
    python tiny_aes.py --dataset_path "~/datasets/tiny_aes_sedpack/" \
                       --original_files "~/datasets/tinyaes"
"""
import argparse
from pathlib import Path
from typing import Any

import keras
import numpy as np
from tqdm import tqdm

from scaaml.metrics.custom import MeanRank
from scaaml.models import get_gpam_model
from sedpack.io import (
    Dataset,
    DatasetFillerContext,
    Metadata,
    DatasetStructure,
    Attribute,
)
from sedpack.io.types import SplitT


def add_shard(shard_file: Path, dataset_filler: DatasetFillerContext,
              split: SplitT) -> tuple[float, float]:
    # Reading these one by one is slow.
    shard = np.load(shard_file)
    traces = np.squeeze(shard["traces"], axis=2)
    keys = np.transpose(shard["keys"])
    ciphertexts = np.transpose(shard["cts"])
    sub_bytes_out = np.transpose(shard["sub_bytes_out"])
    sub_bytes_in = np.transpose(shard["sub_bytes_in"])
    plaintexts = np.transpose(shard["pts"])

    # Running extremes
    running_min: float = 1e10
    running_max: float = -1e10

    for i in range(256):
        # Update the extremes
        running_min = np.min(traces[i])
        running_max = np.max(traces[i])

        values = {
            "trace1": traces[i],
            "key": keys[i],
            "ciphertext": ciphertexts[i],
            "sub_bytes_out": sub_bytes_out[i],
            "sub_bytes_in": sub_bytes_in[i],
            "plaintext": plaintexts[i],
        }
        dataset_filler.write_example(
            values=values,
            split=split,
            custom_metadata={
                # For performance reasons it is better to group
                # write_example calls with the same metadata.
                "key": shard["keys"][:, i].tolist(),
            },
        )

    return running_min, running_max


def convert_to_sedpack(dataset_path: Path, original_files: Path) -> None:
    """Convert the tinyAES dataset from the original format to sedpack.

    Args:

      dataset_path (Path): The newly created sedpack dataset.

      original_files (Path): Path to the original NumPy files.
    """
    # Make sure that the original files are present.
    test_dir: Path = original_files / "test"
    train_dir: Path = original_files / "train"
    if not test_dir.is_dir() or not train_dir.is_dir():
        raise FileNotFoundError("Expected the original NumPy files to be in "
                                f"{test_dir} and {train_dir}")
    test_files: list[Path] = list(test_dir.glob("*.npz"))
    train_files: list[Path] = list(train_dir.glob("*.npz"))
    if not test_files or not train_files:
        raise FileNotFoundError(f"Expected .npz files in {test_dir} and "
                                f"{train_dir}.")

    # General information about the dataset.
    metadata = Metadata(
        description="SCAAML AES side-channel attacks tutorial",
        dataset_license="Apache License 2.0",
        custom_metadata={
            "purpose":
                "For educational and demo purpose only",
            "implementation":
                "TinyAES",
            "cite":
                """
            @inproceedings{burszteindc27,
            title={A Hacker Guide To Deep Learning Based Side Channel Attacks},
            author={Elie Bursztein and Jean-Michel Picod},
            booktitle ={DEF CON 27},
            howpublished = {\\url{https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/}}
            year={2019},
            editor={DEF CON}
            }
            """,  # pylint: disable=line-too-long
            "original from":
                "https://github.com/google/scaaml/tree/main/scaaml_intro",
        },
    )

    # Types of attributes stored
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(
                name="trace1",
                shape=(80_000,),
                dtype="float16",
            ),
            Attribute(name="key", shape=(16,), dtype="uint8"),
            Attribute(name="plaintext", shape=(16,), dtype="uint8"),
            Attribute(name="sub_bytes_out", shape=(16,), dtype="uint8"),
            Attribute(name="sub_bytes_in", shape=(16,), dtype="uint8"),
            Attribute(name="ciphertext", shape=(16,), dtype="uint8"),
        ],
        shard_file_type="fb",
        compression="LZ4",
    )

    # Create a new dataset
    dataset = Dataset.create(
        path=dataset_path,  # All files are stored here
        metadata=metadata,
        dataset_structure=dataset_structure,
    )

    # Running extremes
    running_min: float = 1e10
    running_max: float = -1e10

    # Fill in the examples.
    with dataset.filler() as dataset_filler:
        # Arbitrary split of the test files into test and holdout (otherwise we
        # have no holdout). Even files into "holdout".
        for shard_file in tqdm(test_files[::2], desc="Fill the holdout split"):
            shard_min, shard_max = add_shard(
                shard_file=shard_file,
                dataset_filler=dataset_filler,
                split="holdout",
            )
            running_min = min(running_min, shard_min)
            running_max = max(running_max, shard_max)

        # Odd files into "test".
        for shard_file in tqdm(test_files[1::2], desc="Fill the test split"):
            shard_min, shard_max = add_shard(
                shard_file=shard_file,
                dataset_filler=dataset_filler,
                split="test",
            )
            running_min = min(running_min, shard_min)
            running_max = max(running_max, shard_max)

        for shard_file in tqdm(train_files, desc="Fill the train split"):
            shard_min, shard_max = add_shard(
                shard_file=shard_file,
                dataset_filler=dataset_filler,
                split="train",
            )
            running_min = min(running_min, shard_min)
            running_max = max(running_max, shard_max)

    # Update extremes of the trace (first attribute).
    dataset.dataset_structure.saved_data_description[0].custom_metadata.update({
        "min": float(running_min),
        "max": float(running_max)
    })
    dataset.write_config()


def process_record(record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Processing of a single record. The input is a dictionary of string and
    tensor, the output of this function is a tuple the neural network's input
    (trace) and a dictionary of one-hot encoded expected outputs.
    """
    # The first neural network was using just the first half of the trace:
    inputs = record["trace1"]
    outputs = {
        "sub_bytes_in_0":
            keras.ops.one_hot(
                record["sub_bytes_in"][0],
                num_classes=256,
            ),
    }
    return (inputs, outputs)


def train(dataset_path: Path) -> None:
    """Train a GPAM model on the given dataset.
    """
    batch_size: int = 64  # hyperparameter
    steps_per_epoch: int = 800  # hyperparameter
    epochs: int = 750  # hyperparameter
    target_lr: float = 0.0005  # hyperparameter
    merge_filter_1: int = 0  # hyperparameter
    merge_filter_2: int = 0  # hyperparameter
    trace_len: int = 80_000  # hyperparameter
    patch_size: int = 200  # hyperparameter
    val_steps: int = 16

    # Load the dataset
    dataset = Dataset(dataset_path)

    # Create the definition of inputs and outputs.
    trace_min = dataset.dataset_structure.saved_data_description[
        0].custom_metadata["min"]
    trace_max = dataset.dataset_structure.saved_data_description[
        0].custom_metadata["max"]
    inputs = {"trace1": {"min": trace_min, "delta": trace_max - trace_min}}
    outputs = {"sub_bytes_in_0": {"max_val": 256}}

    model = get_gpam_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=[],
        trace_len=trace_len,
        merge_filter_1=merge_filter_1,
        merge_filter_2=merge_filter_2,
        patch_size=patch_size,
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adafactor(target_lr),
        loss=["categorical_crossentropy" for _ in range(len(outputs))],
        metrics={name: ["acc", MeanRank()] for name in outputs},
    )
    model.summary()

    train_ds = dataset.as_tfdataset(
        split="train",
        process_record=process_record,
        batch_size=batch_size,
        #file_parallelism=4,
        #parallelism=4,
    )
    validation_ds = dataset.as_tfdataset(
        split="test",
        process_record=process_record,
        batch_size=batch_size,
        #file_parallelism=4,
        #parallelism=4,
    )

    # Train the model.
    _ = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=val_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_path",
                        "-d",
                        help="Where to save the dataset",
                        type=Path,
                        required=True)
    parser.add_argument("--original_files",
                        "-o",
                        help="Original NumPy files path (optional)",
                        type=Path)
    args = parser.parse_args()

    if args.original_files:
        convert_to_sedpack(
            dataset_path=args.dataset_path,
            original_files=args.original_files,
        )

    train(dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
