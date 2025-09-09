from enum import Enum
from pprint import pp
from typing import Annotated

from cstructimpl import *


def test_basic_usage():
    class TwoFields(CStruct):
        a: Annotated[int, CType.U8]
        b: Annotated[int, CType.U8]

    assert TwoFields.c_size() == 2
    assert TwoFields.c_align() == 1
    assert TwoFields.c_build(bytes([1, 2])) == TwoFields(1, 2)


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
