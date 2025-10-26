import pytest


@pytest.fixture(scope="session", autouse=True)
def import_cstructimpl_dep(doctest_namespace):
    from cstructimpl import (
        CStruct,
        CArray,
        CBool,
        CFloat,
        CInt,
        CMapper,
        CPadding,
        CStr,
        Autocast,
        BaseType,
    )
    from typing import Annotated
    from enum import Enum, IntEnum

    doctest_namespace.update(locals())
