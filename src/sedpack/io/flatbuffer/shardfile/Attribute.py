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

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: shardfile

# pylint: skip-file

import flatbuffers  # type: ignore[import-untyped]
from flatbuffers.compat import import_numpy  # type: ignore[import-untyped]

import numpy as np
import numpy.typing as npt


class Attribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf: bytes, offset: int = 0) -> "Attribute":
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Attribute()
        x.Init(buf, n + offset)
        return x

    # Attribute
    def Init(self, buf: bytes, pos: int) -> None:
        self._tab = flatbuffers.table.Table(buf, pos)

    # Attribute
    def AttributeBytes(self, j: int) -> bytes:
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(  # type: ignore[no-any-return]
                flatbuffers.number_types.Uint8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return bytes([])

    # Attribute
    def AttributeBytesAsNumpy(self) -> npt.NDArray[np.uint8]:
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(  # type: ignore[no-any-return]
                flatbuffers.number_types.Uint8Flags, o)
        return np.array([], dtype=np.uint8)

    # Attribute
    def AttributeBytesLength(self) -> int:
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)  # type: ignore[no-any-return]
        return 0

    # Attribute
    def AttributeBytesIsNone(self) -> bool:
        o: int = flatbuffers.number_types.UOffsetTFlags.py_type(
            self._tab.Offset(4))
        return o == 0


def AttributeStart(  # type: ignore[no-any-unimported]
    builder: flatbuffers.builder.Builder) -> None:
    builder.StartObject(1)


def Start(  # type: ignore[no-any-unimported]
    builder: flatbuffers.builder.Builder) -> None:
    AttributeStart(builder)


def AttributeAddAttributeBytes(  # type: ignore[no-any-unimported]
        builder: flatbuffers.builder.Builder, attributeBytes: int) -> None:
    builder.PrependUOffsetTRelativeSlot(
        0,
        flatbuffers.number_types.UOffsetTFlags.py_type(attributeBytes),
        0,
    )


def AddAttributeBytes(  # type: ignore[no-any-unimported]
        builder: flatbuffers.builder.Builder, attributeBytes: int) -> None:
    AttributeAddAttributeBytes(builder, attributeBytes)


def AttributeStartAttributeBytesVector(  # type: ignore[no-any-unimported]
        builder: flatbuffers.builder.Builder, numElems: int) -> int:
    return builder.StartVector(1, numElems, 1)  # type: ignore[no-any-return]


def StartAttributeBytesVector(  # type: ignore[no-any-unimported]
        builder: flatbuffers.builder.Builder, numElems: int) -> int:
    return AttributeStartAttributeBytesVector(builder, numElems)


def AttributeEnd(  # type: ignore[no-any-unimported]
    builder: flatbuffers.builder.Builder) -> int:
    return builder.EndObject()  # type: ignore[no-any-return]


def End(  # type: ignore[no-any-unimported]
    builder: flatbuffers.builder.Builder) -> int:
    return AttributeEnd(builder)
