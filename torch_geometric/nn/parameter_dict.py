from typing import Iterable, Mapping, Optional

import torch
from torch.nn import Parameter


# `torch.nn.ParameterDict` doesn't allow `.` to be used in key names.
# This `ParameterDict` will support it by converting the `.` to `#` in the
# internal representation and converts it back to `.` in the external
# representation.
class ParameterDict(torch.nn.ParameterDict):
    def __init__(self, parameters: Optional[Mapping[str, Parameter]] = None):
        # Replace the keys in modules.
        if parameters:
            parameters = {
                self.to_internal_key(key): module
                for key, module in parameters.items()
            }
        super().__init__(parameters)

    @staticmethod
    def to_internal_key(key: str) -> str:
        return key.replace(".", "#")

    @staticmethod
    def to_external_key(key: str) -> str:
        return key.replace("#", ".")

    def __getitem__(self, key: str) -> Parameter:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: str, parameter: Parameter) -> None:
        return super().__setitem__(self.to_internal_key(key), parameter)

    def __delitem__(self, key: str) -> None:
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: str) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[str]:
        return [self.to_external_key(key) for key in super().keys()]
