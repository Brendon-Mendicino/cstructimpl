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
        byteorder: Literal["little", "big"] = "little",
        signed: bool | None = None,
    ) -> T | None: ...
```

Any class that follows this protocol can act as a `BaseType`, controlling its own parsing, size, and alignment.

When parsing a struct:

- If a field type is itself a `BaseType`, parsing happens automatically.  
- Otherwise, annotate the field with `Annotated[..., BaseType]` to tell the parser how to interpret it.

The library comes with a set of ready-to-use type definitions that cover the majority of C primitive types.

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
    kind: Annotated[ResultType, CBuilder(CType.U8, lambda u8: ResultType(u8))]
    error_code: Annotated[int, CType.I32]
```

But much simpler and less error-prone.

---

## ✨ Features

- Define Python classes that map directly to C `struct`s
- Parse raw bytes into typed objects with a single method call
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
