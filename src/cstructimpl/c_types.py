import sys

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Callable,
    Generic,
    Literal,
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
        while True:
            batch = tuple(islice(iterable, n, None))
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

    def c_signed(self) -> bool: ...

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> T | None: ...

    def c_encode(
        self,
        data: T,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
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

    def c_signed(self) -> bool:
        return self.has_ctype.c_get_type().c_signed()

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ):
        return self.has_ctype.c_get_type().c_build(
            raw, is_little_endian=is_little_endian, signed=signed
        )

    def c_encode(
        self,
        data: HasBaseType,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> bytes:
        return self.has_ctype.c_get_type().c_encode(
            data, is_little_endian=is_little_endian, signed=signed
        )


class CType(Enum):
    """Represents the C native types."""

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

    def c_signed(self) -> bool:
        match self:
            case self.I8 | self.I16 | self.I32 | self.I64 | self.I128:
                return True

            case self.U8 | self.U16 | self.U32 | self.U64 | self.U128:
                return False

            case _:
                raise RuntimeError(
                    f"Should not be here! {self=} Type was not supported: the match was not exaustive."
                )

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> int:
        if signed is None:
            signed = self.c_signed()

        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        return int.from_bytes(
            raw, byteorder="little" if is_little_endian else "big", signed=signed
        )

    def c_encode(
        self, data: int, *, is_little_endian: bool = True, signed: bool | None = None
    ) -> bytes:
        if signed is None:
            signed = self.c_signed()

        return data.to_bytes(
            length=self.c_size(),
            byteorder="little" if is_little_endian else "big",
            signed=signed,
        )


@dataclass
class CArray(Generic[T], BaseType[list[T]]):
    """Represents a generic sized array."""

    ctype: BaseType[T]
    array_size: int

    def c_size(self) -> int:
        return self.ctype.c_size() * self.array_size

    def c_align(self) -> int:
        return self.ctype.c_align()

    def c_signed(self) -> bool:
        raise NotImplementedError()

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> list[T]:
        _ = signed

        if len(raw) != self.c_size():
            raise ValueError(
                f"The raw bytes did not have the same lenght of the type! {self=} {len(raw)=}"
            )

        return [
            cast(
                T,
                self.ctype.c_build(
                    bytes(cell_bytes), is_little_endian=is_little_endian
                ),
            )
            for cell_bytes in batched(raw, self.ctype.c_size())
        ]


@dataclass
class CPadding(BaseType[None]):
    """Represent padding bytes between the actual values."""

    padding: int

    def c_size(self) -> int:
        return self.padding

    def c_align(self) -> int:
        return self.padding

    def c_signed(self) -> bool:
        raise NotImplementedError("This method should not be called!")

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> None:
        _ = raw, is_little_endian, signed

        return None

    def c_encode(
        self, data: None, *, is_little_endian: bool = True, signed: bool | None = None
    ) -> bytes:
        _ = data, is_little_endian, signed

        return int(0).to_bytes(self.c_size())


@dataclass
class CStr(BaseType[str]):
    """Represents C string with a null-termination character."""

    array_size: int
    align: int = 1
    encoding: str = "utf-8"

    def c_size(self) -> int:
        return self.array_size

    def c_align(self) -> int:
        return self.align

    def c_signed(self) -> bool:
        raise NotImplementedError("This method should not be called!")

    def c_build(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
        signed: bool | None = None,
    ) -> str:
        _ = is_little_endian, signed

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

    def c_encode(
        self, data: str, *, is_little_endian: bool = True, signed: bool | None = None
    ) -> bytes:
        _ = is_little_endian, signed

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
    """Builds a generic object starting from a `BaseType`."""

    ctype: BaseType[U]
    decoder: Callable[[U | None], T]
    encoder: Callable[[T], U]

    def c_size(self) -> int:
        return self.ctype.c_size()

    def c_align(self) -> int:
        return self.ctype.c_align()

    def c_signed(self) -> bool:
        return self.ctype.c_signed()

    def c_build(
        self, raw: bytes, *, is_little_endian: bool = True, signed: bool | None = None
    ) -> T:
        return self.decoder(
            self.ctype.c_build(
                raw,
                is_little_endian=is_little_endian,
                signed=signed,
            )
        )

    def c_encode(
        self, data: T, *, is_little_endian: bool = True, signed: bool | None = None
    ) -> bytes:
        return self.ctype.c_encode(
            self.encoder(data),
            is_little_endian=is_little_endian,
            signed=signed,
        )
