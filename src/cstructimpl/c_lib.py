import sys

from abc import ABC, abstractmethod
from dataclasses import Field, dataclass, fields, is_dataclass
from itertools import islice
from typing import Callable, Generic, Literal, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = object

from .c_types import BaseType, HasBaseType, CPadding

T = TypeVar("T")


def _get_origin(t: type) -> type:
    return getattr(t, "__origin__", t)


def _get_metadata(t: type) -> tuple | None:
    return getattr(t, "__metadata__", None)


def _get_ctype(t: Field) -> BaseType | None:
    origin = _get_origin(t.type)
    metadata = _get_metadata(t.type) or tuple()

    # The metadata can override the base type if requested
    for t in metadata:
        if isinstance(t, HasBaseType):
            return t.c_get_type()

        if isinstance(t, BaseType):
            return t

    # Metadata has precedence over the origin type
    if isinstance(origin, BaseType):
        return origin

    if isinstance(origin, HasBaseType):
        return origin.c_get_type()

    return None


def _types_from_dataclass(cls: type) -> list[BaseType]:
    ctypes = list[BaseType]()

    for field in fields(cls):
        ctype = _get_ctype(field)

        if ctype is None:
            raise ValueError(
                f"The field of the class is not annotated with a Type, nor the orgigin is a Type! {cls=} {field=}"
            )

        ctypes.append(ctype)

    return ctypes


@dataclass
class _Pipeline:
    pipeline: list[BaseType]
    size: int
    align: int


@dataclass
class _StructTypeHandler(Generic[T]):
    """StructTypeHandler"""

    pipeline: _Pipeline
    cls: type[T]

    def c_size(self) -> int:
        return self.pipeline.size

    def c_align(self) -> int:
        return self.pipeline.align

    def c_signed(self) -> bool:
        raise NotImplementedError()

    def c_build(
        self,
        raw: bytes,
        *,
        byteorder: Literal["little", "big"] = "little",
        signed: bool | None = None,
    ) -> T:
        # TODO: handle byteorder, signed, size, align

        raw_slice = islice(raw, None)
        cls_items = []

        for pipe_item in self.pipeline.pipeline:
            raw_bytes = islice(raw_slice, pipe_item.c_size())

            cls_item = pipe_item.c_build(
                bytes(raw_bytes),
                byteorder=byteorder,
                signed=signed,
            )

            if cls_item is not None:
                cls_items.append(cls_item)

        return self.cls(*cls_items)


@dataclass
class _UnionTypeHandler(Generic[T]):
    """StructTypeHandler"""

    pipeline: _Pipeline
    cls: type[T]

    def c_size(self) -> int:
        return self.pipeline.size

    def c_align(self) -> int:
        return self.pipeline.align

    def c_signed(self) -> bool:
        raise NotImplementedError()

    def c_build(
        self,
        raw: bytes,
        *,
        byteorder: Literal["little", "big"] = "little",
        signed: bool | None = None,
    ) -> T:
        # TODO: handle byteorder, signed, size, align

        cls_items = []

        for pipe_item in self.pipeline.pipeline:
            raw_bytes = islice(raw, pipe_item.c_size())

            cls_item = pipe_item.c_build(
                bytes(raw_bytes),
                byteorder=byteorder,
                signed=signed,
            )

            if cls_item is not None:
                cls_items.append(cls_item)

        return self.cls(*cls_items)


def _build_struct_pipeline(
    ctypes: list[BaseType], *, override_align: int | None = None
):
    pipeline = list[BaseType]()
    current_size = 0
    current_align = 0

    for ctype in ctypes:
        padding = -current_size % ctype.c_align()

        if padding != 0:
            pipeline.append(CPadding(padding))

        current_align = max(current_align, ctype.c_align())
        current_size += ctype.c_size()
        pipeline.append(ctype)

    if override_align is not None:
        current_align = override_align

    # A struct always needs to always have a size which is a mutiple of its
    # alignemt
    current_size += -current_size % current_align

    return _Pipeline(pipeline, current_size, current_align)


def _c_struct(cls: type[T], align: int | None) -> type[T]:
    if not is_dataclass(cls):
        cls = dataclass(cls)

    if not is_dataclass(cls):
        raise ValueError(
            f"{cls=} is not a dataclass! {cls=} must be a dataclass in order to use c_struct."
        )

    ctypes = _types_from_dataclass(cls)

    pipeline = _build_struct_pipeline(ctypes, override_align=align)

    @classmethod
    def c_get_type(self):
        _ = self
        return _StructTypeHandler(pipeline, cls)

    setattr(cls, "c_get_type", c_get_type)

    return cls


def _c_union(cls: type[T], align: int | None) -> type[T]:
    if not is_dataclass(cls):
        cls = dataclass(cls)

    if not is_dataclass(cls):
        raise ValueError(
            f"{cls=} is not a dataclass! {cls=} must be a dataclass in order to use c_struct."
        )

    ctypes = _types_from_dataclass(cls)
    size = max(map(lambda c: c.c_size(), ctypes), default=0)
    if align is None:
        align = max(map(lambda c: c.c_align(), ctypes), default=0)

    pipeline = _Pipeline(
        pipeline=ctypes,
        size=size,
        align=align,
    )

    @classmethod
    def c_get_type(self):
        _ = self
        return _UnionTypeHandler(pipeline, cls)

    setattr(cls, "c_get_type", c_get_type)

    return cls


def c_struct(*, align: int | None = None, union: bool = False) -> Callable[[T], T]:
    def c_struct_inner(cls: type[T]):
        if not union:
            return _c_struct(cls, align)
        else:
            return _c_union(cls, align)

    return c_struct_inner


class CStruct(ABC):
    """Overrides the attributes of a class, directly embedding the
    method for a BaseType iside the type definition.

    This is a simple wrapper over the `c_struct` decorator."""

    def __init_subclass__(cls, **kwargs):
        new_cls = c_struct(**kwargs)(cls)
        assert isinstance(new_cls, HasBaseType)
        base_type = new_cls.c_get_type()

        if sys.version_info >= (3, 12):
            attrs = BaseType.__protocol_attrs__
        else:
            attrs = {"c_size", "c_align", "c_signed", "c_build"}

        for attr in attrs:
            setattr(cls, attr, getattr(base_type, attr))

    @classmethod
    @abstractmethod
    def c_size(cls) -> int: ...

    @classmethod
    @abstractmethod
    def c_align(cls) -> int: ...

    @classmethod
    @abstractmethod
    def c_signed(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def c_build(
        cls,
        raw: bytes,
        *,
        byteorder: Literal["little", "big"] = "little",
        signed: bool | None = None,
    ) -> Self: ...
