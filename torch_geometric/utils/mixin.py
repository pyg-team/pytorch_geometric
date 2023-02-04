from typing import Iterator


class CastMixin:
    @classmethod
    def cast(cls, *args, **kwargs):  # TODO Can we apply this recursively?
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CastMixin):
                return elem
            if isinstance(elem, tuple):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.values())
