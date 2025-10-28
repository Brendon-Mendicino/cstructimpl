import sys
from typing import Callable, Concatenate, Generic, Iterable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class hybridmethod(Generic[T, P, R_co]):
    # """Like `@classmethod` but if called on the instance of a class,
    # the reference to the object is passed to the first
    # parameter of the method instead of the class instance.

    # # Example

    # >>> class

    # """

    def __init__(self, f: Callable[Concatenate[type[T] | T, P], R_co], /) -> None:
        self.f = f

    @property
    def __func__(self) -> Callable[Concatenate[type[T] | T, P], R_co]:
        return self.f

    def __get__(self, instance: T | None, owner: type[T]) -> Callable[P, R_co]:
        if instance is None:
            return lambda *args, **kwargs: self.f(owner, *args, **kwargs)
        else:
            return lambda *args, **kwargs: self.f(instance, *args, **kwargs)

    if sys.version_info >= (3, 10):

        @property
        def __wrapped__(self) -> Callable[Concatenate[type[T] | T, P], R_co]:
            return self.f


class peekable(Generic[T]):
    """An iterable class which can return the next element of the wrapped
    iterator without advancing it."""

    def __init__(self, iterable: Iterable[T]):
        self.head = None
        self.iterator = iter(iterable)

    def peek(self):
        if self.head is None:
            self.head = next(self.iterator, None)
            return self.head
        else:
            return self.head

    def __next__(self):
        if self.head is not None:
            retval = self.head
            self.head = None
            return retval
        else:
            return next(self.iterator)

    def __iter__(self):
        return self
