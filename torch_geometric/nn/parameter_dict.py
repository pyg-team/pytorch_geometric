from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
from torch.nn import Parameter

Key = Union[str, Tuple[str, ...]]


# `torch.nn.ParameterDict` doesn't allow `.` to be used in key names.
# This `ParameterDict` will support it by converting the `.` to `#` in the
# internal representation and converts it back to `.` in the external
# representation. It also allows passing tuples as keys.
class ParameterDict(torch.nn.ParameterDict):
    def __init__(
        self,
        parameters: Optional[Mapping[Key, Parameter]] = None,
    ):
        # Replace the keys in modules.
        if parameters:
            parameters = {
                self.to_internal_key(key): parameter
                for key, parameter in parameters.items()
            }
        super().__init__(parameters)

    @classmethod
    def to_internal_key(cls, key: Key) -> str:
        # ParameterDict cannot handle tuples as keys:
        if isinstance(key, tuple):
            assert len(key) > 1
            key = f"<{'___'.join(key)}>"
        assert isinstance(key, str)

        # ParameterDict cannot handle keys that exists as class attributes:
        if hasattr(cls, key):
            key = f'<{key}>'

        # ParameterDict cannot handle dots in keys:
        return key.replace('.', '#')

    @classmethod
    def to_external_key(cls, key: str) -> Key:
        key = key.replace('#', '.')

        if key[0] == '<' and key[-1] == '>' and hasattr(cls, key[1:-1]):
            key = key[1:-1]

        if key[0] == '<' and key[-1] == '>' and '___' in key:
            key = tuple(key[1:-1].split('___'))

        return key

    def __getitem__(self, key: Key) -> Parameter:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: Key, parameter: Parameter):
        return super().__setitem__(self.to_internal_key(key), parameter)

    def __delitem__(self, key: Key):
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: Key) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[Key]:
        return [self.to_external_key(key) for key in super().keys()]

    def items(self) -> Iterable[Tuple[Key, Parameter]]:
        return [(self.to_external_key(k), v) for k, v in super().items()]
