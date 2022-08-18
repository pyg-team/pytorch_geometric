from typing import Iterable, Mapping, Optional

from torch.nn import Module, ModuleDict


# `torch.nn.ModuleDict` doesn't allow `.` to be used in the keys.
# `CustomModuleDict` will support it by converting the `.` to `#` in the
# internal representation and converts it back to `.` in the external
# representation.
class CustomModuleDict(ModuleDict):
    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        # Replace the keys in modules.
        if modules:
            modules = {
                CustomModuleDict.to_internal_key(key): module
                for key, module in modules.items()
            }
        super().__init__(modules)

    @staticmethod
    def to_internal_key(key: str) -> str:
        return key.replace(".", "#")

    @staticmethod
    def to_external_key(key: str) -> str:
        return key.replace("#", ".")

    def __getitem__(self, key: str) -> Module:
        return super().__getitem__(CustomModuleDict.to_internal_key(key))

    def __setitem__(self, key: str, module: Module) -> None:
        return super().__setitem__(CustomModuleDict.to_internal_key(key),
                                   module)

    def __delitem__(self, key: str) -> None:
        return super().__delitem__(CustomModuleDict.to_internal_key(key))

    def __contains__(self, key: str) -> bool:
        return super().__contains__(CustomModuleDict.to_internal_key(key))

    def keys(self) -> Iterable[str]:
        return [
            CustomModuleDict.to_external_key(key) for key in super().keys()
        ]
