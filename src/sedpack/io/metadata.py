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
"""Metadata of a dataset."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

import sedpack
from sedpack.io.types import CompressionT, HashChecksumT, ShardFileTypeT, SplitT
from sedpack.io.shard_file_metadata import ShardListInfo


class Metadata(BaseModel):
    """High level descriptions of data.

    Attributes:

        description (str): Short human-readable description of this dataset.

        dataset_license (str): License of the data and usage.

        dataset_version (str): Version of this dataset.

        download_from (str): Download URL for where this data has been
        downloaded. Details TBD when download API is decided.

        custom_metadata (dict[str, Any]): Custom metadata. Needs to be
        serializable as JSON.

        sedpack_version (str): Version of the dataset library used to
        create this dataset. Auto-filled.
    """
    description: str = ""
    dataset_license: str = "https://creativecommons.org/licenses/by/4.0/"
    dataset_version: str = "1.0.0"
    download_from: str = ""
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    sedpack_version: str = sedpack.__version__


class Attribute(BaseModel):
    """Description of a single attribute of an example.

    Attributes:

        name (str): Name of this attribute. Must be unique across all
        attributes of a single example.

        dtype (str): Description which type is used. If dtype is "bytes" and
        shape is an empty tuple checking shape is omitted to allow saving
        arbitrary length file contents (e.g., JPEG files provide much better
        compression than generic compression algorithms).

        shape (tuple[int, ...]): Shape of the saved data. Empty tuple stands
        for an unknown shape but then `dtype` must be "bytes". Only strictly
        positive integers.

        custom_metadata (dict[str, Any]): Custom metadata. Needs to be
        serializable as JSON.
    """
    name: str
    dtype: str
    shape: tuple[int, ...]
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("shape")
    @classmethod
    def only_positive_dimensions(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Allow only positive dimensions in a shape. Namely there cannot be a
        value of -1 or None which are sometimes used in reshaping.

        Args:

          cls: A BaseModel has to be a classmethod.

          v (tuple[int, ...]): The shape dimensions.

        Returns: unchanged value `v`.

        Raises: ValueError when there is any dimension in `v` which is not a
        positive integer.
        """
        if any(dimension <= 0 for dimension in v):
            raise ValueError("All dimensions in the shape must be strictly "
                             "positive")
        return v

    def has_variable_size(self) -> bool:
        """Determine if this attribute can have a variable size. Only true when
        the dtype is either "bytes" or "str" and shape is empty tuple.
        """
        return self.dtype in ["bytes", "str"] and self.shape == ()


class DatasetStructure(BaseModel):
    """How data are saved and represented.

    Attributes:

        saved_data_description (list[Attribute]): Description of all saved
        attributes. When `shard_file_type` is "fb" then all attributes must be
        saved in this order.

        examples_per_shard (int): How many examples are saved in a single shard
        file.

        compression (str): Which compression to use.

        shard_file_type (ShardFileTypeT): Which file-type is used to store
        shard information.

        hash_checksum_algorithms (tuple[HashChecksumT, ...]): Which hash
        algorithms should be computed for file hash checksums.
    """
    saved_data_description: list[Attribute] = Field(default_factory=list)
    compression: CompressionT = "GZIP"
    examples_per_shard: int = 256
    shard_file_type: ShardFileTypeT = "tfrec"
    hash_checksum_algorithms: tuple[HashChecksumT, ...] = ("sha256",)


class DatasetInfo(BaseModel):
    """Holds all information saved in the main metadata file
    dataset_info.json.

    Attributes:

      metadata (Metadata): Dataset metadata.

      dataset_structure (DatasetStructure): Structure of saved data.

      shards_list (dict[SplitT, list[ShardInfo]]): A dictionary with a list of
      all ShardInfo in a given split.

      splits (dict[SplitT, list[ShardInfo]]): A dictionary with a list of top
      level `ShardInfo` in a given split.
    """
    metadata: Metadata = Field(default_factory=Metadata)
    dataset_structure: DatasetStructure = Field(
        default_factory=DatasetStructure)
    splits: dict[SplitT, ShardListInfo] = Field(default_factory=dict)
