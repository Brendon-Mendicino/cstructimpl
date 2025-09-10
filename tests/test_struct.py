from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from cstructimpl import *


def test_basic_decoding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CType.U8]
        y: Annotated[int, CType.U8]

    assert Point.c_size() == 2
    assert Point.c_align() == 1
    assert Point.c_build(bytes([1, 2])) == Point(1, 2)


def test_basic_deconding_with_padding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CType.U8]
        y: Annotated[int, CType.U16]

    assert Point.c_size() == 4
    assert Point.c_align() == 2
    assert Point.c_build(bytes([1, 2, 3, 4])) == Point(1, 3 + (4 << 8))


def test_basic_encoding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CType.U8]
        y: Annotated[int, CType.U8]

    assert Point(1, 2).c_encode() == bytes([1, 2])


def test_encoding_with_padding():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CType.U8]
        y: Annotated[int, CType.U16]

    assert Point(1, (255 << 8) + 2).c_encode() == bytes([1, 0, 2, 255])


def test_encoding_with_default_value():
    @dataclass
    class Point(CStruct):
        x: Annotated[int, CType.U8]
        y: Annotated[int, CType.U8] = 2

    assert Point(1).c_encode() == bytes([1, 2])


def test_embedded_struct():
    class Inner(CStruct):
        a: Annotated[int, CType.U8]
        b: Annotated[int, CType.U8]

    class Outer(CStruct):
        a: Annotated[int, CType.U16]
        inner: Inner

    assert Outer.c_size() == 4
    assert Outer.c_align() == 2
    assert Outer.c_build(bytes([1, 0, 2, 3])) == Outer(1, Inner(2, 3))


def test_struct_with_string():
    class SWithStr(CStruct):
        size: Annotated[int, CType.U16]
        string: Annotated[str, CStr(5)]

    assert SWithStr.c_build(bytes([5, 0]) + b"Helo\x00") == SWithStr(5, "Helo")


def test_autocast_with_enums():
    class PersonType(Enum):
        HAPPY = 0
        SAD = 1

    class Person(CStruct):
        age: Annotated[int, CType.U16]
        person: Annotated[PersonType, CType.U8, Autocast()]

    assert Person.c_build(bytes([18, 0, 1, 0])) == Person(18, PersonType.SAD)


def test_struct_with_lists():
    @dataclass
    class Inner(CStruct, align=2):
        first: Annotated[int, CType.U8]
        second: Annotated[int, CType.U8]
        third: Annotated[int, CType.U8]

    @dataclass
    class MyList(CStruct):
        list: Annotated[list[Inner], CArray(Inner, 3)]

    data = bytes(range(1, 13))  # 3 items × 4 bytes each
    parsed = MyList.c_build(data)

    assert parsed == MyList(
        [
            Inner(1, 2, 3),
            Inner(5, 6, 7),
            Inner(9, 10, 11),
        ]
    )


def test_custom_defined_base_type():
    @dataclass
    class UnixTimestamp:
        def c_size(self) -> int:
            return 4

        def c_align(self) -> int:
            return 4

        def c_signed(self) -> bool:
            return False

        def c_build(
            self,
            raw: bytes,
            *,
            is_little_endian: bool = True,
            signed=False,
        ) -> datetime:
            ts = int.from_bytes(
                raw, byteorder="little" if is_little_endian else "big", signed=signed
            )
            return datetime.fromtimestamp(ts)

        def c_encode(self):
            pass

    @dataclass
    class LogEntry(CStruct):
        timestamp: Annotated[datetime, UnixTimestamp()]
        level: Annotated[int, CType.U8]

    parsed = LogEntry.c_build(bytes([255, 0, 0, 0, 3, 0, 0, 0]))
    assert parsed == LogEntry(datetime.fromtimestamp(255), 3)
