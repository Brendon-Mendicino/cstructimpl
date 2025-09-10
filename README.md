# `cstructimpl`

> A Python package for translating C `struct`s into Python classes.

[![PyPI version](https://img.shields.io/pypi/v/cstructimpl.svg)](https://pypi.org/project/cstructimpl/)
[![License](https://img.shields.io/github/license/Brendon-Mendicino/cstructimpl.svg)](https://github.com/Brendon-Mendicino/cstructimpl/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/cstructimpl.svg)](https://pypi.org/project/cstructimpl/)

---

## ⚡ Quick Start

Install from PyPI:

```bash
pip install cstructimpl
```

Define your struct and parse raw bytes:

```python
from cstructimpl import *


class Info(CStruct):
    age: Annotated[int, CType.U8]
    height: Annotated[int, CType.U8]


class Person(CStruct):
    info: Info
    name: Annotated[str, CStr(6)]


person = Person.c_build(bytes([18, 170]) + b"Pippo\x00")
print(person)  # Person(info=Info(age=18, height=170), name='Pippo')
```

---

## 🚀 Introduction

`cstructimpl` makes working with binary data in Python simple and intuitive.  
By subclassing `CStruct`, you can define Python classes that map directly to C-style `struct`s and parse raw bytes into fully typed objects.

No manual parsing, no boilerplate — just define your struct and let the library do the heavy lifting.

---

## 🔧 Type System

At the core of the library is the `BaseType` protocol, which defines how types behave in the C world:

```python
class BaseType(Protocol[T]):

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
```

Any class that follows this protocol can act as a `BaseType`, controlling its own parsing, size, and alignment.

When parsing a struct:

- If a field type is itself a `BaseType`, parsing happens automatically.  
- Otherwise, annotate the field with `Annotated[..., BaseType]` to tell the parser how to interpret it.

The library comes with a set of ready-to-use type definitions that cover the majority of C primitive types.

---

## 📌 Examples

Here are a few practical examples showing how `cstructimpl` works in real-world scenarios.

### Basic Deserialization

Define a simple struct with two fields:

```python
class Point(CStruct):
    x: Annotated[int, CType.U8]
    y: Annotated[int, CType.U8]


assert Point.c_size() == 2
assert Point.c_align() == 1
assert Point.c_build(bytes([1, 2])) == Point(1, 2)
```

---

### Serializing a Class

Create a class instance and serlialize it to raw bytes

```python
class Rect(CStruct):
    width: Annotated[int, CType.U8]
    height: Annotated[int, CType.U8] = 10

rect = Rect(2)
assert rect.c_encode() == bytes([2, 10])
```

---

### Nested Structs

You can embed structs inside other structs:

```python
class Dimensions(CStruct):
    width: Annotated[int, CType.U8]
    height: Annotated[int, CType.U8]


class Rectangle(CStruct):
    id: Annotated[int, CType.U16]
    dims: Dimensions


assert Rectangle.c_size() == 4
assert Rectangle.c_align() == 2
assert Rectangle.c_build(bytes([1, 0, 2, 3])) == Rectangle(1, Dimensions(2, 3))
```

---

### Strings in Structs

Support for C-style null-terminated strings:

```python
class Message(CStruct):
    length: Annotated[int, CType.U16]
    text: Annotated[str, CStr(5)]


raw = bytes([5, 0]) + b"Helo\x00"
assert Message.c_build(raw) == Message(5, "Helo")
```

---

### Enums with Autocast

Automatically cast numeric values into Python `Enum`s:

```python
class Mood(Enum):
    HAPPY = 0
    SAD = 1


class Person(CStruct):
    age: Annotated[int, CType.U16]
    mood: Annotated[Mood, CType.U8, Autocast()]


raw = bytes([18, 0, 1, 0])
assert Person.c_build(raw) == Person(18, Mood.SAD)
```

---

### Arrays of Structs

Define fixed-size arrays of structs inside another struct:

```python
class Item(CStruct, align=2):
    a: Annotated[int, CType.U8]
    b: Annotated[int, CType.U8]
    c: Annotated[int, CType.U8]


class ItemList(CStruct):
    items: Annotated[list[Item], CArray(Item, 3)]


data = bytes(range(1, 13))  # 3 items × 4 bytes each
parsed = ItemList.c_build(data)

assert parsed == ItemList([
    Item(1, 2, 3),
    Item(5, 6, 7),
    Item(9, 10, 11),
])
```

### Custom BaseType

> > Hey! Is there a type that serializes an hash-map of list of structs of ...?
> 
> > Yeah, sure there is! You can do it yourself!

`cstructimpl` lets you define your own `BaseType` implementations to handle any kind of data that  is not present among the built-in primitives.

For example, here's a custom type that interprets a raw integer as a **Unix timestamp**, returning a Python `datetime` object:

```python
class UnixTimestamp(BaseType[datetime]):
    def c_size(self) -> int:
        return 4

    def c_align(self) -> int:
        return 4

    def c_signed(self) -> bool:
        return False

    def c_build(self, raw: bytes, *, byteorder="little", signed=False) -> datetime:
        ts = int.from_bytes(raw, byteorder=byteorder, signed=signed)
        return datetime.utcfromtimestamp(ts)
    

    @dataclass
    class LogEntry(CStruct):
        timestamp: Annotated[datetime, UnixTimestamp()]
        level: Annotated[int, CType.U8]


    parsed = LogEntry.c_build(bytes([255, 0, 0, 0, 3, 0, 0, 0]))
    assert parsed == LogEntry(datetime.fromtimestamp(255), 3)
```

---


## 🎭 Autocast

Sometimes raw numeric values carry semantic meaning. In C, this is usually handled with `enum`s.  
With `cstructimpl`, you can automatically reinterpret values into enums (or other types) using `Autocast`.

```python
from cstructimpl import *


class ResultType(Enum):
    OK = 0
    ERROR = 1


class Person(CStruct):
    kind: Annotated[ResultType, CType.U8, Autocast()]
    error_code: Annotated[int, CType.I32]
```

This is equivalent to writing a custom builder:

```python
from cstructimpl import *


class ResultType(Enum):
    OK = 0
    ERROR = 1


class Person(CStruct):
    kind: Annotated[ResultType, CMapper(CType.U8, lambda u8: ResultType(u8))]
    error_code: Annotated[int, CType.I32]
```

But much simpler and less error-prone.

---

## ✨ Features

- Define Python classes that map directly to C `struct`s
- Parse raw bytes into typed objects with a single method call
- Serialize a class to raw bytes using built-in type system
- Built-in type system for common C primitives
- Support for nested structs
- Flexible extension via the `BaseType` protocol

---

## 📖 Use Cases

- Parsing binary network protocols  
- Working with binary file formats  
- Interfacing with C libraries and data structures  
- Replacing boilerplate parsing code with clean, type-safe classes  

---

## 📚 Documentation

More detailed usage examples and advanced topics are available in the [documentation](https://github.com/Brendon-Mendicino/cstructimpl/wiki).

---

## 🤝 Contributing

Contributions are welcome!  

If you'd like to improve `cstructimpl`, please open an issue or submit a pull request on [GitHub](https://github.com/Brendon-Mendicino/cstructimpl).

---

## 📜 License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/Brendon-Mendicino/cstructimpl/blob/main/LICENSE).
