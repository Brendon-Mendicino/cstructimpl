from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum, IntFlag
import math
from pprint import pp
import struct
from typing import Annotated

import pytest

from cstructimpl import *


def test_basic_decoding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U8]
        y: Annotated[int, CInt.U8]

    assert Point.c_size() == 2
    assert Point.c_align() == 1
    assert Point.c_decode(bytes([1, 2])) == Point(1, 2)


def test_basic_deconding_with_padding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U8]
        y: Annotated[int, CInt.U16]

    assert Point.c_size() == 4
    assert Point.c_align() == 2
    assert Point.c_decode(bytes([1, 2, 3, 4])) == Point(1, 3 + (4 << 8))


def test_basic_encoding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U8]
        y: Annotated[int, CInt.U8]

    assert Point(1, 2).c_encode() == bytes([1, 2])


def test_encoding_with_padding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U8]
        y: Annotated[int, CInt.U16]

    assert Point.c_size() == 4
    assert Point.c_align() == 2
    assert Point(1, (255 << 8) + 2).c_encode() == bytes([1, 0, 2, 255])


def test_encoding_with_padding_at_the_end():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U16]
        y: Annotated[int, CInt.U8]

    assert Point(1, 2).c_encode() == bytes([1, 0, 2, 0])


def test_encoding_with_default_value():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CInt.U8]
        y: Annotated[int, CInt.U8] = 2

    assert Point(1).c_encode() == bytes([1, 2])


def test_default_basetypes():
    @dataclass
    class Mixed(CStruct):
        x: int
        y: bool
        z: float

    assert Mixed.c_decode(
        bytes([2, 0, 0, 0, 1, 0, 0, 0]) + bytes.fromhex("00007F43")
    ) == Mixed(2, True, 255.0)


@pytest.mark.parametrize(
    "val", [0.0, 1.0, -3.14, 9999.123, float("inf"), float("-inf")]
)
def test_float_value(val: float):
    @dataclass
    class Rect(CStruct):
        width: Annotated[float, CFloat.F32]
        height: Annotated[float, CFloat.F64]

    inc = 1e200
    val = float(val)
    rect = Rect(val, val * inc)

    assert rect.c_encode() == struct.pack("<fxxxxd", val, val * inc)


def test_embedded_struct():
    @dataclass
    class Inner(CStruct):
        a: Annotated[int, CInt.U8]
        b: Annotated[int, CInt.U8]

    @dataclass
    class Outer(CStruct):
        a: Annotated[int, CInt.U16]
        inner: Inner

    assert Outer.c_size() == 4
    assert Outer.c_align() == 2
    assert Outer.c_decode(bytes([1, 0, 2, 3])) == Outer(1, Inner(2, 3))


def test_struct_with_string():
    @dataclass
    class SWithStr(CStruct):
        size: Annotated[int, CInt.U16]
        string: Annotated[str, CStr(6)]

    assert SWithStr.c_decode(bytes([5, 0]) + b"Hello\x00") == SWithStr(5, "Hello")


def test_autocast_with_enums():
    class PersonType(Enum):
        HAPPY = 0
        SAD = 1

    @dataclass
    class Person(CStruct):
        age: Annotated[int, CInt.U16]
        person: Annotated[PersonType, CInt.U8, Autocast()]

    assert Person.c_decode(bytes([18, 0, 1, 0])) == Person(18, PersonType.SAD)


@pytest.mark.parametrize("cint", list(CInt))
def test_with_list_of_int(cint: CInt):
    @dataclass
    class Inner(CStruct):
        a: Annotated[int, cint]
        b: Annotated[int, cint]

    @dataclass
    class MyList(CStruct):
        list: Annotated[list[Inner], CArray(Inner.c_get_type(), 3)]

    padding = b"\x00" * (cint.c_size() - 1)
    data = (
        padding.join(i.to_bytes(length=1, byteorder="little") for i in range(1, 7))
        + padding
    )
    print(data)
    my = MyList(list=[Inner(1, 2), Inner(3, 4), Inner(5, 6)])

    assert MyList.c_decode(data) == my
    assert my.c_encode() == data


def test_struct_with_lists_and_custom_align():
    @dataclass
    class Inner(CStruct, align=2):
        first: Annotated[int, CInt.U8]
        second: Annotated[int, CInt.U8]
        third: Annotated[int, CInt.U8]

    @dataclass
    class MyList(CStruct):
        list: Annotated[list[Inner], CArray(Inner.c_get_type(), 3)]

    data = bytes(range(1, 13))  # 3 items × 4 bytes each
    parsed = MyList.c_decode(data)

    assert parsed == MyList(
        [
            Inner(1, 2, 3),
            Inner(5, 6, 7),
            Inner(9, 10, 11),
        ]
    )
    assert parsed.c_encode() == bytes([1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0])


def test_list_interaction_alignment():
    @dataclass
    class ListAlign(CStruct):
        a: Annotated[int, CInt.U16]
        b: Annotated[int, CInt.U8]
        c: Annotated[list[int], CArray(CInt.U8, 3)]

    assert ListAlign.c_size() == 6
    assert ListAlign.c_align() == 2


def test_custom_defined_base_type():
    class UnixTimestamp:
        def c_size(self) -> int:
            return 4

        def c_align(self) -> int:
            return 4

        def c_decode(
            self,
            raw: bytes,
            *,
            is_little_endian: bool = True,
        ) -> datetime:
            byteorder = "little" if is_little_endian else "big"
            ts = int.from_bytes(raw, byteorder=byteorder, signed=False)
            return datetime.fromtimestamp(ts)

        def c_encode(self):
            pass

    @dataclass
    class LogEntry(CStruct):
        timestamp: Annotated[datetime, UnixTimestamp()]
        level: Annotated[int, CInt.U8]

    parsed = LogEntry.c_decode(bytes([255, 0, 0, 0, 3, 0, 0, 0]))
    assert parsed == LogEntry(datetime.fromtimestamp(255), 3)


def test_packed_struct():
    @dataclass
    class Packed(CStruct, packed=True):
        a: Annotated[int, CInt.I8]
        b: Annotated[int, CInt.I32]

    @dataclass
    class Outer(CStruct):
        a: Annotated[int, CInt.I16]
        b: Packed

    assert Packed.c_size() == 5
    assert Packed.c_align() == 1
    assert Outer.c_size() == 8
    assert Outer.c_align() == 2

    raw = bytes([3, 0, 1, 2, 0, 0, 0, 0])
    actual = Outer(3, Packed(1, 2))

    assert Outer.c_decode(raw) == actual
    assert actual.c_encode() == raw


def test_bitfields_basics():
    @dataclass
    class Header(CStruct):
        a: Annotated[int, CInt.U8, BitField(4)]
        b: Annotated[int, CInt.U8, BitField(4)]

    header = Header(1, 2)

    assert Header.c_size() == 1

    assert Header.c_decode(0x21.to_bytes(1, byteorder="little")) == header

    assert (
        Header.c_decode(0x21.to_bytes(1, byteorder="big"), is_little_endian=False)
        == header
    )

    assert header.c_encode() == 0x21.to_bytes(1, byteorder="little", signed=False)


def test_bitfields_inner_paddings():
    @dataclass
    class Header(CStruct):
        a: Annotated[int, CInt.U8, BitField(7)]
        b: Annotated[int, CInt.U8, BitField(4)]
        c: Annotated[int, CInt.U8, BitField(5)]

    assert Header.c_size() == 3
    assert Header.c_align() == 1
    assert Header(0xFF, 0xFF, 0xFF).c_encode() == 0x1F0F7F.to_bytes(
        3, byteorder="little", signed=False
    )


def test_bitfields_decoding():
    @dataclass
    class Header(CStruct):
        a: Annotated[int, CInt.U8, BitField(7)]
        b: Annotated[int, CInt.U8, BitField(4)]
        c: Annotated[int, CInt.U8, BitField(5)]

    assert Header.c_decode(bytes([0xFF, 0xFF, 0xFF])) == Header(0x7F, 0x0F, 0x1F)


def test_bitfields_different_sizes():
    @dataclass
    class Header(CStruct):
        a: Annotated[int, CInt.U16, BitField(1)]
        b: Annotated[int, CInt.U16, BitField(1)]
        c: Annotated[int, CInt.U8, BitField(1)]

    assert Header.c_size() == 4
    assert Header.c_decode(
        0x010002.to_bytes(4, byteorder="little", signed=False)
    ) == Header(0, 1, 1)


def test_bitfields_with_end_marker():
    @dataclass
    class Header(CStruct):
        a: Annotated[int, CInt.U16, BitField(1)]
        b: Annotated[int, CInt.U16, BitField(1, True)]
        c: Annotated[int, CInt.U16, BitField(1)]

    assert Header.c_size() == 4
    assert Header.c_decode(
        (2 | (1 << 2 * 8)).to_bytes(4, byteorder="little", signed=False)
    ) == Header(0, 1, 1)


def test_bitfields_with_autocasting():
    class Kind(IntEnum):
        A = 0
        B = 1
        C = 2
        D = 3

    class Option(IntFlag):
        X = 1 << 0
        Y = 1 << 1
        W = 1 << 2
        Z = 1 << 3

    @dataclass
    class Header(CStruct):
        a: Annotated[Kind, CInt.U8, BitField(4), Autocast()]
        b: Annotated[Option, CInt.U8, BitField(4), Autocast()]

    raw = 0x62.to_bytes(1, byteorder="little", signed=False)
    assert Header.c_size() == 1
    assert Header.c_decode(raw) == Header(Kind.C, Option.Y | Option.W)
