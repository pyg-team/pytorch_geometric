from typing import Any, Iterator, List, Mapping, Tuple


class MappingView:
    def __init__(self, mapping: Mapping[str, Any], *args: str):
        self._mapping = mapping
        self._args = args

    def _keys(self) -> List[str]:
        if len(self._args) == 0:
            return list(self._mapping.keys())
        else:
            return [arg for arg in self._args if arg in self._mapping]

    def __len__(self) -> int:
        return len(self._keys())

    def __repr__(self) -> str:
        mapping = {key: self._mapping[key] for key in self._keys()}
        return f'{self.__class__.__name__}({mapping})'

    __class_getitem__ = classmethod(type([]))  # type: ignore


class KeysView(MappingView):
    def __iter__(self) -> Iterator[str]:
        yield from self._keys()


class ValuesView(MappingView):
    def __iter__(self) -> Iterator[Any]:
        for key in self._keys():
            yield self._mapping[key]


class ItemsView(MappingView):
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for key in self._keys():
            yield (key, self._mapping[key])
