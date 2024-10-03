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

Example use:
    python tiny_aes.py --dataset_path "~/datasets/tiny_aes_sedpack/" --original_files "~/datasets/tinyaes"
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sedpack.io import Dataset, DatasetFiller, Metadata, DatasetStructure, Attribute


def add_shard(shard_file: Path, dataset_filler: DatasetFiller,
              split: str) -> None:
    # Reading these one by one is slow.
    shard = np.load(shard_file)
    traces = np.squeeze(shard["traces"], axis=2)
    keys = np.transpose(shard["keys"])
    cts = np.transpose(shard["cts"])
    sub_bytes_out = np.transpose(shard["sub_bytes_out"])
    sub_bytes_in = np.transpose(shard["sub_bytes_in"])
    plaintexts = np.transpose(shard["pts"])

    for i in range(256):
        values = {
            "trace": traces[i],
            "key": keys[i],
            "ct": cts[i],
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


def create_dataset(dataset_path: Path, original_files: Path) -> None:
    # Make sure that the original files are present.
    test_dir: Path = original_files / "test"
    train_dir: Path = original_files / "train"
    if not test_dir.is_dir() or not train_dir.is_dir():
        print
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
            """,
            "original from":
                "https://github.com/google/scaaml/tree/main/scaaml_intro",
        },
    )

    # Types of attributes stored
    dataset_structure = DatasetStructure(
        saved_data_description=[
            Attribute(
                name="trace",
                shape=(80_000,),
                dtype="float16",
            ),
            Attribute(name="key", shape=(16,), dtype="uint8"),
            Attribute(name="plaintext", shape=(16,), dtype="uint8"),
            Attribute(name="sub_bytes_out", shape=(16,), dtype="uint8"),
            Attribute(name="sub_bytes_in", shape=(16,), dtype="uint8"),
            Attribute(name="ct", shape=(16,), dtype="uint8"),
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

    # Fill in the examples.
    with dataset.filler() as dataset_filler:
        for shard_file in tqdm(test_files, desc="Fill the test split"):
            add_shard(
                shard_file=shard_file,
                dataset_filler=dataset_filler,
                split="test",
            )

        for shard_file in tqdm(train_files, desc="Fill the train split"):
            add_shard(
                shard_file=shard_file,
                dataset_filler=dataset_filler,
                split="train",
            )


def train(dataset_path: Path) -> None:
    # TODO
    pass


def main():
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
        create_dataset(
            dataset_path=args.dataset_path,
            original_files=args.original_files,
        )

    train()


if __name__ == "__main__":
    main()
