import struct
import sys

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Callable,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)
from itertools import islice

if sys.version_info >= (3, 12):
    from itertools import batched
    from typing import Self
else:

    def batched(iterable, n: int):
        iterator = iter(iterable)
        while True:
            batch = tuple(islice(iterator, n))
            if len(batch) == 0:
                break

            yield batch

    Self = object


T = TypeVar("T")
U = TypeVar("U")


@runtime_checkable
class BaseType(Protocol[T]):
    """Protocol specification to parse a raw bytes into a
    structure."""

    def c_size(self) -> int: ...

    def c_align(self) -> int: ...

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> T | None: ...

    def c_encode(
        self,
        data: T,
        *,
        is_little_endian: bool = True,
    ) -> bytes: ...


@runtime_checkable
class HasBaseType(Protocol):
    @classmethod
    def c_get_type(cls) -> BaseType[Self]: ...


@dataclass
class GetType:
    """Extracts a ctype from a class type."""

    has_ctype: HasBaseType

    def c_size(self) -> int:
        return self.has_ctype.c_get_type().c_size()

    def c_align(self) -> int:
        return self.has_ctype.c_get_type().c_align()

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ):
        return self.has_ctype.c_get_type().c_decode(
            raw,
            is_little_endian=is_little_endian,
        )

    def c_encode(
        self,
        data: HasBaseType,
        *,
        is_little_endian: bool = True,
    ) -> bytes:
        return self.has_ctype.c_get_type().c_encode(
            data,
            is_little_endian=is_little_endian,
        )


class CInt(Enum):
    """Represents the C native int types.

    Args:

    Examples:

    >>> class Point(CStruct):
    ...     x: Annotated[int, CInt.U16]
    ...     y: Annotated[int, CInt.U8]
    ...
    >>> Point.c_decode(bytes([1, 0, 2, 0]))
    Point(x=1, y=2)

    """

    I8 = auto()
    U8 = auto()
    I16 = auto()
    U16 = auto()
    I32 = auto()
    U32 = auto()
    I64 = auto()
    U64 = auto()
    I128 = auto()
    U128 = auto()

    @classmethod
    def get_unsigned(cls, size: int):
        match size:
            case 1:
                return cls.U8
            case 2:
                return cls.U16
            case 4:
                return cls.U32
            case 8:
                return cls.U64
            case 16:
                return cls.U128
            case _:
                raise ValueError(f"{size=} is not valid!")

    def c_size(self) -> int:
        match self:
            case self.I8 | self.U8:
                return 1

            case self.I16 | self.U16:
                return 2

            case self.I32 | self.U32:
                return 4

            case self.I64 | self.U64:
                return 8

            case self.I128 | self.U128:
                return 16

            case _:
                raise RuntimeError(
                    f"Should not be here! {self=} Type was not supported: the match was not exaustive."
                )

    def c_align(self) -> int:
        return self.c_size()

    def _signed(self) -> bool:
        match self:
            case self.I8 | self.I16 | self.I32 | self.I64 | self.I128:
                return True

            case self.U8 | self.U16 | self.U32 | self.U64 | self.U128:
                return False

            case _:
                raise RuntimeError(
                    f"Should not be here! {self=} Type was not supported: the match was not exaustive."
                )

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> int:
        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        return int.from_bytes(
            raw,
            byteorder="little" if is_little_endian else "big",
            signed=self._signed(),
        )

    def c_encode(self, data: int, *, is_little_endian: bool = True) -> bytes:
        return data.to_bytes(
            length=self.c_size(),
            byteorder="little" if is_little_endian else "big",
            signed=self._signed(),
        )


class CBool(BaseType[bool]):
    """Represent a C bool value.

    Args:

    Examples:

    >>> class Person(CStruct):
    ...     age: int
    ...     adult: Annotated[bool, CBool()]
    >>> raw = bytes([23, 0, 0, 0, 1, 0, 0, 0])
    >>> Person.c_decode(raw)
    Person(age=23, adult=True)

    """

    def c_size(self) -> int:
        return 1

    def c_align(self) -> int:
        return 1

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> bool:
        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        byteorder = "little" if is_little_endian else "big"

        return bool.from_bytes(raw, byteorder=byteorder)

    def c_encode(
        self,
        data: bool,
        *,
        is_little_endian: bool = True,
    ) -> bytes:
        byteorder = "little" if is_little_endian else "big"

        return data.to_bytes(length=self.c_size(), byteorder=byteorder)


class CFloat(Enum):
    """Conforms to the IEEE 754 standard of encoded floating point numbers.

    Args:

    Examples:

    >>> import struct
    >>> class Point(CStruct):
    ...     x: float
    ...     y: float
    >>> raw = struct.pack("<ff", 1.0, 22.0)
    >>> Point.c_decode(raw)
    Point(x=1.0, y=22.0)

    """

    F32 = auto()
    F64 = auto()

    def _float_fmt(self):
        match self:
            case self.F32:
                return "f"

            case self.F64:
                return "d"

            case _:
                raise RuntimeError(
                    f"Should not be here! {self=} Type was not supported: the match was not exaustive."
                )

    def c_size(self) -> int:
        match self:
            case self.F32:
                return 4

            case self.F64:
                return 8

            case _:
                raise RuntimeError(
                    f"Should not be here! {self=} Type was not supported: the match was not exaustive."
                )

    def c_align(self) -> int:
        return self.c_size()

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> float:
        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        align = "<" if is_little_endian else ">"

        unpack = struct.unpack(align + self._float_fmt(), raw)

        return unpack[0]

    def c_encode(
        self,
        data: float,
        *,
        is_little_endian: bool = True,
    ) -> bytes:
        align = "<" if is_little_endian else ">"

        return struct.pack(align + self._float_fmt(), data)


@dataclass
class CArray(Generic[T], BaseType[list[T]]):
    """Represents a generic sized array.

    Args:

    Examples:

    >>> class Stream(CStruct):
    ...     id: int
    ...     values: Annotated[list[int], CArray(CInt.U8, 4)]
    >>> raw = bytes([20, 0, 0, 0, 1, 2, 3, 4])
    >>> Stream.c_decode(raw)
    Stream(id=20, values=[1, 2, 3, 4])

    """

    ctype: BaseType[T]
    array_size: int

    def c_size(self) -> int:
        return self.ctype.c_size() * self.array_size

    def c_align(self) -> int:
        return self.ctype.c_align()

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> list[T]:
        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        return [
            cast(
                T,
                self.ctype.c_decode(
                    bytes(cell_bytes), is_little_endian=is_little_endian
                ),
            )
            for cell_bytes in batched(raw, self.ctype.c_size())
        ]

    def c_encode(
        self,
        data: list[T],
        *,
        is_little_endian: bool = True,
    ) -> bytes:
        if len(data) != self.array_size:
            raise ValueError(
                f"The length of the array is different from the one spcified in the CArray field! {len(data)=} {self.array_size=}"
            )

        return b"".join(
            self.ctype.c_encode(item, is_little_endian=is_little_endian)
            for item in data
        )


@dataclass
class CPadding(BaseType[None]):
    """Represent padding bytes between the actual values.

    Args:

    Examples:

    >>> class Point(CStruct):
    ...     x: Annotated[int, CInt.U8]
    ...     padding: Annotated[None, CPadding(1)]
    ...     y: Annotated[int, CInt.U16]
    >>> raw = bytes([1, 2, 3, 0])
    >>> Point.c_decode(raw)
    Point(x=1, padding=None, y=3)

    """

    padding: int

    def c_size(self) -> int:
        return self.padding

    def c_align(self) -> int:
        return self.padding

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> None:
        _ = raw, is_little_endian

        return None

    def c_encode(self, data: None, *, is_little_endian: bool = True) -> bytes:
        _ = data, is_little_endian

        return int(0).to_bytes(self.c_size(), byteorder="little", signed=False)


@dataclass
class CStr(BaseType[str]):
    """Represents C string with a null-termination character.

    Args:

    Examples:

    >>> class Person(CStruct):
    ...     age: int
    ...     name: Annotated[str, CStr(8)]
    >>> raw = bytes([18, 0, 0, 0]) + b'Peppino\\x00'
    >>> Person.c_decode(raw)
    Person(age=18, name='Peppino')

    """

    array_size: int
    align: int = 1
    encoding: str = "utf-8"

    def c_size(self) -> int:
        return self.array_size

    def c_align(self) -> int:
        return self.align

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> str:
        _ = is_little_endian

        if len(raw) != self.array_size:
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        # Find null-terminator
        null_index = raw.find(b"\x00")
        if null_index == -1:
            raise ValueError(
                f"The null-terminator was not found while parsing a string. {self=} {raw=}"
            )

        return bytes(islice(raw, null_index)).decode(self.encoding)

    def c_encode(self, data: str, *, is_little_endian: bool = True) -> bytes:
        _ = is_little_endian

        encoded = data.encode(encoding=self.encoding) + b"\x00"
        # Fill the ramining bytes with zero values
        encoded += b"\x00" * (self.c_size() - len(encoded))

        if len(encoded) != self.c_size():
            raise ValueError(
                f"Failed to encode a str! The lenght of the string is greater than the actual size it can hold! (Remember a CStr must be null-terminated) {len(encoded)=} {self.c_size()=}"
            )

        return encoded


@dataclass
class CMapper(Generic[T, U], BaseType[T]):
    """Builds a generic object starting from a `BaseType`.

    Args:

    Examples:

    >>> from enum import Enum
    >>> class ResultType(Enum):
    ...     OK = 0
    ...     ERROR = 1
    >>> class Message(CStruct):
    ...    kind: Annotated[ResultType, CMapper(CInt.U8, ResultType, lambda r: r.value)]
    ...    error_code: Annotated[int, CInt.I32]
    >>> raw = bytes([1, 0, 0, 0, 23, 0, 0, 0])
    >>> Message.c_decode(raw)
    Message(kind=<ResultType.ERROR: 1>, error_code=23)
    >>> Message(ResultType.ERROR, 65).c_encode()
    b'\\x01\\x00\\x00\\x00A\\x00\\x00\\x00'

    """

    ctype: BaseType[U]
    decoder: Callable[[U | None], T]
    encoder: Callable[[T], U]

    def c_size(self) -> int:
        return self.ctype.c_size()

    def c_align(self) -> int:
        return self.ctype.c_align()

    def c_decode(self, raw: bytes, *, is_little_endian: bool = True) -> T:
        return self.decoder(self.ctype.c_decode(raw, is_little_endian=is_little_endian))

    def c_encode(self, data: T, *, is_little_endian: bool = True) -> bytes:
        return self.ctype.c_encode(
            self.encoder(data), is_little_endian=is_little_endian
        )


@dataclass
class _MarkerBitField(BaseType[T]):
    """BitField marker class."""

    inner: BaseType[T]
    cint: CInt
    field_size: int
    end_marker: bool

    def c_size(self) -> int:
        return self.inner.c_size()

    def c_align(self) -> int:
        return self.inner.c_align()

    def c_decode(self, raw: bytes, *, is_little_endian: bool = True) -> T | None:
        return self.inner.c_decode(raw, is_little_endian=is_little_endian)

    def c_encode(self, data: T, *, is_little_endian: bool = True) -> bytes:
        return self.inner.c_encode(data, is_little_endian=is_little_endian)
