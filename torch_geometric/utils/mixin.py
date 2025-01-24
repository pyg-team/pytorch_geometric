from typing import Any, Iterator, TypeVar

T = TypeVar('T')


class CastMixin:
    @classmethod
    def cast(cls: T, *args: Any, **kwargs: Any) -> T:
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None  # type: ignore
            if isinstance(elem, CastMixin):
                return elem  # type: ignore
            if isinstance(elem, tuple):
                return cls(*elem)  # type: ignore
            if isinstance(elem, dict):
                return cls(**elem)  # type: ignore
        return cls(*args, **kwargs)  # type: ignore

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.values())
