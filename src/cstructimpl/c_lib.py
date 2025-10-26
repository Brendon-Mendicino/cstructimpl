"""Decorators and classes to extend types."""

import inspect
import sys

from dataclasses import Field, dataclass, fields, is_dataclass
from itertools import islice
from typing import (
    Callable,
    TypeVar,
    get_origin,
)

from .util import hybridmethod
from .c_annotations import Autocast
from .c_types import BaseType, CBool, CMapper, HasBaseType, CPadding, CType, CFloat

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = object

T = TypeVar("T")


DEFAULT_TYPE_TO_BASETYPE = {
    bool: CBool(),
    int: CType.I32,
    float: CFloat.F32,
}


class _MarkerPadding(CPadding):
    pass


def _get_origin(t: type) -> type:
    origin = getattr(t, "__origin__", t)
    return get_origin(origin) or origin


def _get_field_origin(f: Field) -> type:
    return _get_origin(f.type)


def _get_metadata(t: type) -> tuple | None:
    return getattr(t, "__metadata__", None)


def _get_field_metadata(f: Field) -> tuple | None:
    return _get_metadata(f.type)


def _is_autocast(f: Field) -> Autocast | None:
    autocast = _get_field_metadata(f) or tuple()
    autocast = filter(lambda f: isinstance(f, Autocast), autocast)
    return next(autocast, None)


def _get_ctype_decode_type(ctype: BaseType[T]) -> type[T]:
    ret_type = inspect.signature(ctype.c_decode).return_annotation
    ret_type = _get_origin(ret_type)
    return ret_type


def _get_default_basetype(t: type) -> BaseType | None:
    return DEFAULT_TYPE_TO_BASETYPE.get(t, None)


def _get_basetype(t: Field) -> BaseType | None:
    """Get the `BaseType` of a `Field`."""

    origin = _get_field_origin(t)
    metadata = _get_field_metadata(t) or tuple()

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

    # If no metadata is provided try and check if the current
    # type has a default `BaseType`
    return _get_default_basetype(origin)


def _types_from_dataclass(cls: type) -> list[BaseType]:
    """Extracts the BaseType from each dataclass field."""
    base_types = list[BaseType]()

    for field in fields(cls):
        autocast = _is_autocast(field)
        base_type = _get_basetype(field)

        if base_type is None:
            raise ValueError(
                f"The field of the class is not annotated with a BaseType, nor the origin is a BaseType! {cls=} {field=}"
            )

        if autocast:
            cls_field_type = _get_field_origin(field)
            ret_type = _get_ctype_decode_type(base_type)

            if not isinstance(cls_field_type, type):
                raise ValueError(
                    f"Autocast is set to True, but the dataclass field is not an instance of `type`! Cannot cast serialized object to the field type. Either disable autocast or change type signature. {cls_field_type=}"
                )

            base_type = CMapper(base_type, cls_field_type, ret_type)

        base_types.append(base_type)

    return base_types


def _strict_dataclass_fields_check(cls: type, cls_items: list):
    for field, item in zip(fields(cls), cls_items):
        ftype = _get_field_origin(field)

        if not isinstance(item, ftype):
            raise ValueError(
                f"Cannot assign value of type {type(item)=} to dataclass field {field.name=} of type {field.type=}"
            )


@dataclass
class _Pipeline:
    pipeline: list[BaseType]
    size: int
    align: int


@dataclass
class _StructTypeHandler(BaseType[T]):
    """StructTypeHandler"""

    pipeline: _Pipeline
    cls: type[T]
    strict: bool

    def c_size(self) -> int:
        return self.pipeline.size

    def c_align(self) -> int:
        return self.pipeline.align

    def c_decode(self, raw: bytes, *, is_little_endian: bool = True) -> T:
        # TODO: handle byteorder, signed, size, align

        raw_slice = islice(raw, None)
        cls_items = []

        for pipe_item in self.pipeline.pipeline:
            raw_bytes = bytes(islice(raw_slice, pipe_item.c_size()))

            if isinstance(pipe_item, _MarkerPadding):
                continue

            cls_item = pipe_item.c_decode(raw_bytes, is_little_endian=is_little_endian)

            cls_items.append(cls_item)

        if self.strict:
            _strict_dataclass_fields_check(self.cls, cls_items)

        return self.cls(*cls_items)

    def c_encode(self, data: T, *, is_little_endian: bool = True) -> bytes:
        raw_data = bytes()
        data_fields = (getattr(data, f.name) for f in fields(data))

        for pipe_item in self.pipeline.pipeline:
            if isinstance(pipe_item, _MarkerPadding):
                raw_data += pipe_item.c_encode(None, is_little_endian=is_little_endian)
                continue

            raw_data += pipe_item.c_encode(
                next(data_fields), is_little_endian=is_little_endian
            )

        return raw_data


@dataclass
class _UnionTypeHandler(BaseType[T]):
    """StructTypeHandler"""

    pipeline: _Pipeline
    cls: type[T]
    strict: bool

    def c_size(self) -> int:
        return self.pipeline.size

    def c_align(self) -> int:
        return self.pipeline.align

    def c_decode(
        self,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> T:
        # TODO: handle byteorder, signed, size, align

        cls_items = []

        for pipe_item in self.pipeline.pipeline:
            raw_bytes = islice(raw, pipe_item.c_size())

            cls_item = pipe_item.c_decode(
                bytes(raw_bytes), is_little_endian=is_little_endian
            )

            if cls_item is not None:
                cls_items.append(cls_item)

        if self.strict:
            _strict_dataclass_fields_check(self.cls, cls_items)

        # TODO: make union fields lazyly evalutated
        return self.cls(*cls_items)

    def c_encode(self, data: T, *, is_little_endian: bool = True) -> bytes:
        raise NotImplementedError()


def _build_struct_pipeline(
    ctypes: list[BaseType], *, override_align: int | None = None
):
    pipeline = list[BaseType]()
    current_size = 0
    current_align = 0

    for ctype in ctypes:
        padding = -current_size % ctype.c_align()

        if padding != 0:
            pipeline.append(_MarkerPadding(padding))
            current_size += padding

        current_align = max(current_align, ctype.c_align())
        current_size += ctype.c_size()
        pipeline.append(ctype)

    # Override struct global padding
    if override_align is not None:
        current_align = override_align

    # Add trailing padding if needed
    padding = -current_size % current_align
    if padding != 0:
        pipeline.append(_MarkerPadding(padding))

    # A struct always needs to always have a size which is a mutiple of its
    # alignemt
    current_size += -current_size % current_align

    return _Pipeline(pipeline, current_size, current_align)


def _c_struct(cls: type[T], align: int | None, strict: bool) -> type[T]:
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
        return _StructTypeHandler(pipeline, cls, strict)

    setattr(cls, "c_get_type", c_get_type)

    return cls


def _c_union(cls: type[T], align: int | None, strict: bool) -> type[T]:
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
        return _UnionTypeHandler(pipeline, cls, strict)

    setattr(cls, "c_get_type", c_get_type)

    return cls


def c_struct(
    *,
    align: int | None = None,
    union: bool = False,
    strict: bool = True,
) -> Callable[[type[T]], type[T]]:
    """Decorator used to automatically implement the `BaseType`
    methods in a class. Every class that is annotated with
    this decorator, automatically becomes a `BaseType` itself
    by providing the `c_get_type()` method.

    This decorator converts the class in a `dataclass` it isn't already,
    this is done in order to exploit the dataclasses utility
    methods for the fields of the class.

    Args:
        align (int | None, optional): struct alignment. Defaults to None.
        union (bool, optional): if true the fields of a struct are
            interpreted as a C union. Defaults to False.
        strict (bool, optional): check that the converted value in
            a parameter matches the actual type defined in the class. Defaults to True.

    Returns:
        _ (Callable[[type[T]], type[T]]): returns the augmented class

    Examples:

    >>> @c_struct()
    ... class Point:
    ...     x: int
    ...     y: int
    >>> point_ctype = Point.c_get_type()
    >>> p = point_ctype.c_decode(bytes([1, 0, 0, 0, 2, 0, 0, 0]))
    >>> assert p == Point(1, 2)
    """

    def c_struct_inner(cls: type[T]):
        if not union:
            return _c_struct(cls, align, strict)
        else:
            return _c_union(cls, align, strict)

    return c_struct_inner


class CStruct:
    """Overrides the attributes of a class, directly embedding the
    method for a `BaseType` iside the type definition.

    This is a simple wrapper over the `c_struct` decorator.

    Args:

    Examples:

    >>> class Point(CStruct):
    ...     x: int
    ...     y: int
    >>> p = Point.c_decode(bytes([1, 0, 0, 0, 2, 0, 0, 0]))
    >>> assert p == Point(1, 2)
    """

    def __init_subclass__(cls, **kwargs):
        new_cls = c_struct(**kwargs)(cls)
        assert isinstance(new_cls, HasBaseType)
        base_type = new_cls.c_get_type()

        if sys.version_info >= (3, 12):
            attrs = BaseType.__protocol_attrs__
        else:
            attrs = {"c_size", "c_align", "c_signed", "c_decode", "c_encode"}

        for attr in attrs:
            if attr == "c_encode":
                setattr(cls, "_c_encode", getattr(base_type, attr))
            else:
                setattr(cls, attr, getattr(base_type, attr))

    @classmethod
    def c_size(cls) -> int: ...

    @classmethod
    def c_align(cls) -> int: ...

    @classmethod
    def c_decode(
        cls,
        raw: bytes,
        *,
        is_little_endian: bool = True,
    ) -> Self: ...

    # TODO: decide if this method should be split into two separate ones, one left as the signature of `BaseType` and another one that class this with data=self
    @hybridmethod
    def c_encode(
        self,
        data: Self | None = None,
        *,
        is_little_endian: bool = True,
    ) -> bytes:
        if data is None and not isinstance(self, type):
            data = self

        if data is None:
            raise ValueError("Data is None!")

        return self._c_encode(data, is_little_endian=is_little_endian)

    @classmethod
    def c_get_type(cls) -> BaseType[Self]: ...
