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

import flatbuffers
import numpy as np

from sedpack.io.metadata import Attribute
from sedpack.io.shard.shard_writer_flatbuffer import ShardWriterFlatBuffer
from sedpack.io.flatbuffer.iterate import IterateShardFlatBuffer

from sedpack.io.flatbuffer.unit_tests.shard_writer_flatbuffer_test_schema.NumPyVectorTest import *


def test_can_write_and_read():
    """Write using ShardWriterFlatBuffer and read using generated FlatBuffers
    code.
    """
    # Write a FlatBuffer.
    builder = flatbuffers.Builder(0)

    attributes: dict[str, np.ndarray] = {
        "attribute_bool":
            np.random.randint(0, 1, size=13, dtype=bool),
        "attribute_byte":
            np.random.randint(-120, 120, size=17, dtype=np.int8),
        "attribute_ubyte":
            np.random.randint(0, 256, size=23, dtype=np.uint8),
        #// 16 bit
        "attribute_short":
            np.random.randint(-1000, 1000, size=7, dtype=np.int16),
        "attribute_ushort":
            np.random.randint(0, 1000, size=(3, 7), dtype=np.uint16),

        #// 32 bit
        "attribute_int":
            np.random.randint(-2000, 2000, size=(5, 11, 51), dtype=np.int32),
        "attribute_uint":
            np.random.randint(0, 3000, size=(71, 3), dtype=np.uint32),
        "attribute_float":
            np.random.uniform(size=11).astype(dtype=np.float32),

        #// 64 bit
        "attribute_long":
            np.random.randint(-2000, 200, size=(2, 51), dtype=np.int64),
        "attribute_ulong":
            np.random.randint(0, 3333, size=(3, 71), dtype=np.uint64),
        "attribute_double":
            np.random.uniform(size=(11, 91)).astype(dtype=np.float64),
    }

    attribute_offsets: dict[str, int] = {}
    attribute_description: dict[str, Attribute] = {}

    for name, value in reversed(attributes.items()):
        description = Attribute(name=name,
                                shape=value.shape,
                                dtype=str(value.dtype))
        attribute_offsets[
            name] = ShardWriterFlatBuffer.save_numpy_vector_as_bytearray(
                builder,
                attribute=description,
                value=value,
            )
        attribute_description[name] = description

    # Fill
    NumPyVectorTestStart(builder)
    AddAttributeBool(builder, attribute_offsets["attribute_bool"])
    AddAttributeByte(builder, attribute_offsets["attribute_byte"])
    AddAttributeUbyte(builder, attribute_offsets["attribute_ubyte"])
    AddAttributeShort(builder, attribute_offsets["attribute_short"])
    AddAttributeUshort(builder, attribute_offsets["attribute_ushort"])
    AddAttributeInt(builder, attribute_offsets["attribute_int"])
    AddAttributeUint(builder, attribute_offsets["attribute_uint"])
    AddAttributeFloat(builder, attribute_offsets["attribute_float"])
    AddAttributeLong(builder, attribute_offsets["attribute_long"])
    AddAttributeUlong(builder, attribute_offsets["attribute_ulong"])
    AddAttributeDouble(builder, attribute_offsets["attribute_double"])
    test_vectors = NumPyVectorTestEnd(builder)
    builder.Finish(test_vectors)
    byte_buffer = builder.Output()

    # Read back using FlatBuffers API
    parsed_test_vectors = NumPyVectorTest.GetRootAs(byte_buffer)

    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeBoolAsNumpy(),
            attribute=attribute_description["attribute_bool"],
        ), attributes["attribute_bool"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeByteAsNumpy(),
            attribute=attribute_description["attribute_byte"],
        ), attributes["attribute_byte"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeUbyteAsNumpy(),
            attribute=attribute_description["attribute_ubyte"],
        ), attributes["attribute_ubyte"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeShortAsNumpy(),
            attribute=attribute_description["attribute_short"],
        ), attributes["attribute_short"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeUshortAsNumpy(),
            attribute=attribute_description["attribute_ushort"],
        ), attributes["attribute_ushort"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeIntAsNumpy(),
            attribute=attribute_description["attribute_int"],
        ), attributes["attribute_int"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeUintAsNumpy(),
            attribute=attribute_description["attribute_uint"],
        ), attributes["attribute_uint"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeFloatAsNumpy(),
            attribute=attribute_description["attribute_float"],
        ), attributes["attribute_float"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeLongAsNumpy(),
            attribute=attribute_description["attribute_long"],
        ), attributes["attribute_long"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeUlongAsNumpy(),
            attribute=attribute_description["attribute_ulong"],
        ), attributes["attribute_ulong"])
    np.array_equal(
        IterateShardFlatBuffer.decode_array(
            np_bytes=parsed_test_vectors.AttributeDoubleAsNumpy(),
            attribute=attribute_description["attribute_double"],
        ), attributes["attribute_double"])
