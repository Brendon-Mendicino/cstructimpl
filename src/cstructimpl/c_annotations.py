from dataclasses import dataclass


@dataclass
class Autocast:
    """Marker class. Specify if the built object should be autocasted to the dataclass
    actual type.

    >>> from cstructimpl import *
    >>> from enum import IntEnum
    >>> from typing import Annotated
    >>> class Mood(IntEnum):
    ...     HAPPY = 2
    ...     SAD = 3
    >>> class Person(CStruct):
    ...     age: Annotated[int, CInt.U16]
    ...     mood: Annotated[Mood, CInt.U8, Autocast()]
    >>> raw = bytes([50, 0, 3, 0])
    >>> Person.c_decode(raw)
    Person(age=50, mood=<Mood.SAD: 3>)
    >>> list(Person(18, Mood.HAPPY).c_encode())
    [18, 0, 2, 0]
    """

    pass
