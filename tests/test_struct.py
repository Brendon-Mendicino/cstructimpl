from typing import Annotated

from cstructimpl import *


class TwoFields(CStruct):
    a: Annotated[int, CType.U8]
    b: Annotated[int, CType.U8]


def test_basic_usage():
    assert TwoFields.c_size() == 2
    assert TwoFields.c_align() == 1
    assert TwoFields.c_build(bytes([1, 2])) == TwoFields(1, 2)


class Upper(CStruct):
    a: Annotated[int, CType.U16]
    inner: TwoFields


def test_embedded_struct():
    assert Upper.c_size() == 4
    assert Upper.c_align() == 2
    assert Upper.c_build(bytes([1, 0, 2, 3])) == Upper(1, TwoFields(2, 3))
