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
"""Tensorflow utils.

For information how to read and write TFRecord files see
https://www.tensorflow.org/tutorials/load_data/tfrecord
"""

from typing import Any, Callable, cast

import numpy as np
import tensorflow as tf

from sedpack.io.metadata import Attribute


def bytes_feature(value: Any) -> Any:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value: Any) -> Any:
    """Returns a float_list from a float / double."""
    # Fix shape to 1D
    value = tf.constant([value])  # scalar to list, reshaped anyway
    value = tf.reshape(value, -1)
    # Workaround for https://github.com/tensorflow/tensorflow/issues/61671
    value = value.numpy()

    float_list = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=float_list)


def int64_feature(value: Any) -> Any:
    """Returns an int64_list from a bool / enum / int / uint."""
    # Fix shape to 1D
    value = tf.constant([value])  # scalar to list, reshaped anyway
    value = tf.reshape(value, -1)
    # Workaround for https://github.com/tensorflow/tensorflow/issues/61671
    value = value.numpy()

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_from_tfrecord(
        saved_data_description: list[Attribute]) -> Callable[[Any], Any]:
    """Construct the from_tfrecord function.
    """

    # TF_FEATURES construction: must contains all features, it is used in
    # the closure from_tfrecord.

    tf_features = {}  # What is decoded

    # How to decode each attribute
    for attribute in saved_data_description:
        dtype = {
            "str": tf.string,
            "bytes": tf.string,
            "uint8": tf.int64,
            "int8": tf.int64,
            "int16": tf.int64,
            "int32": tf.int64,
            "int64": tf.int64,
            "float16": tf.string,
            "float32": tf.float32,
            "float64": tf.float64,
        }[attribute.dtype]

        shape: tuple[int, ...] = attribute.shape
        if attribute.dtype == "float16":
            # We parse from bytes so no shape
            shape = ()

        tf_features[attribute.name] = tf.io.FixedLenFeature(shape, dtype)

    # Define the decoding function
    # @tf.function
    def from_tfrecord(tf_record: Any) -> Any:
        rec = tf.io.parse_single_example(tf_record, tf_features)
        for attribute in saved_data_description:
            if attribute.dtype == "float16":
                rec[attribute.name] = tf.io.parse_tensor(
                    rec[attribute.name], tf.float16)
                rec[attribute.name] = tf.ensure_shape(rec[attribute.name],
                                                      shape=attribute.shape)
        return rec

    return from_tfrecord


def to_tfrecord(saved_data_description: list[Attribute],
                values: dict[str, Any]) -> bytes:
    """Convert example data into a tfrecord example

    Args:

      saved_data_description (list[Attribute]): Descriptions of all saved
      data.

      values (dict): The name and value to be saved (corresponding to
      saved_data_description).

    Returns: TF.train.Example
    """

    # Check there are no unexpected values
    attribute_names = {attribute.name for attribute in saved_data_description}
    for name in values:
        if name not in attribute_names:
            raise ValueError(f"Unexpected attribute {name} not in "
                             f"ExampleAttributes.")
    # Check there are all expected values
    if len(attribute_names) != len(values):
        raise ValueError(f"There are missing attributes. Got: {values} "
                         f"expected: {attribute_names}")

    # Create dictionary of features
    feature = {}

    for attribute in saved_data_description:
        value = values[attribute.name]

        # Convert the value into a NumPy type.
        value = np.array(value)

        # Check shape
        if attribute.dtype != "bytes" and value.shape != attribute.shape:
            raise ValueError(f"Wrong shape of {attribute.name}, expected: "
                             f"{attribute.shape}, got: {value.shape}.")

        # Set feature value
        if attribute.dtype in ["int8", "uint8", "int32", "int64"]:
            feature[attribute.name] = int64_feature(values[attribute.name])
        elif attribute.dtype == "float16":
            value = value.astype(dtype=np.float16)
            feature[attribute.name] = bytes_feature(
                [tf.io.serialize_tensor(value).numpy()])
        elif attribute.dtype in ["float32", "float64"]:
            feature[attribute.name] = float_feature(values[attribute.name])
        elif attribute.dtype == "str":
            feature[attribute.name] = bytes_feature(
                [values[attribute.name].encode("utf-8")])
        elif attribute.dtype == "bytes":
            feature[attribute.name] = bytes_feature([values[attribute.name]])
        else:
            raise ValueError(f"Unsupported dtype {attribute.dtype} of "
                             f"{attribute.name}.")

    # Return serialized example
    tf_features = tf.train.Features(feature=feature)
    record = tf.train.Example(features=tf_features)
    str_record = cast(bytes, record.SerializeToString())
    return str_record
