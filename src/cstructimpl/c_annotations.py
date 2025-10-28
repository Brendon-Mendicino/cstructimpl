from dataclasses import dataclass


@dataclass
class Autocast:
    """Marker class. Specify if the built object should be autocasted to the dataclass
    actual type.

    >>> from enum import IntEnum
    >>> class Mood(IntEnum):
    ...     NEUTRAL = 0
    ...     HAPPY = 1
    ...     SAD = 2
    ...
    >>> class Person(CStruct):
    ...     age: Annotated[int, CInt.U16]
    ...     mood: Annotated[Mood, CInt.U8, Autocast()]
    ...
    >>> raw = bytes([50, 0, 2, 0])
    >>> Person.c_decode(raw)
    Person(age=50, mood=<Mood.SAD: 2>)
    >>> list(Person(18, Mood.HAPPY).c_encode())
    [18, 0, 1, 0]
    """

    pass


@dataclass
class BitField:
    """Marker class. Declare if a field should be part of bit-field.
    The size in bits must be expressed when annotating, and the
    field can be marked to be the end of a bit-field.

    Args:

    Examples:

    Creating a header class with enums and flags.

    >>> from enum import IntFlag, IntEnum
    >>> class Flags(IntFlag):
    ...     ACK = 1 << 0
    ...     SYN = 1 << 1
    ...     URG = 1 << 2
    ...
    >>> class State(IntEnum):
    ...     PENDING = 0
    ...     ERROR = 1
    ...     SUCCESS = 2
    ...
    >>> class Header(CStruct):
    ...     port: Annotated[int, CInt.U8, BitField(4)]
    ...     id: Annotated[int, CInt.U8, BitField(4)]
    ...     state: Annotated[State, CInt.U8, BitField(2), Autocast()]
    ...     flags: Annotated[Flags, CInt.U8, BitField(3), Autocast()]
    ...     len: Annotated[int, CInt.U8]
    ...
    >>> raw = 0x101A21.to_bytes(3, byteorder="little", signed=False)
    >>> Header.c_decode(raw)
    Header(port=1, id=2, state=<State.SUCCESS: 2>, flags=<Flags.SYN|URG: 6>, len=16)
    """

    field_size: int
    end_marker: bool = False
