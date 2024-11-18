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
"""Build and load tensorFlow dataset Record wrapper"""

from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias  # from typing since 3.10

# Valid split values (used also as directory names).
SplitT: TypeAlias = Literal["train", "test", "holdout"]
TRAIN_SPLIT: SplitT = "train"
TEST_SPLIT: SplitT = "test"
HOLDOUT_SPLIT: SplitT = "holdout"

# Keras model.
TFModelT: TypeAlias = Any

# tf.data.Dataset and similar.
TFDatasetT: TypeAlias = Any

# Type of an attribute value.
AttributeValueT: TypeAlias = Union[str, int, npt.NDArray[np.generic], bytes]

# Compression choices.
CompressionT: TypeAlias = Literal[
    "",
    "BZ2",
    "GZIP",
    "LZMA",
    "LZ4",
    "ZIP",
    "ZLIB",
    "ZSTD",
]

# Hash checksums types (algorithms supported by hashlib).
HashChecksumT: TypeAlias = Literal[
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "xxh32",
    "xxh64",
    "xxh128",
]

# Shard file-type choices. Also serves as the file-extension of shard files.
ShardFileTypeT: TypeAlias = Literal[
    "fb",
    "npz",
    "tfrec",
]

# Type alias for example, this is what gets iterated or saved.
ExampleT: TypeAlias = dict[str, AttributeValueT]
