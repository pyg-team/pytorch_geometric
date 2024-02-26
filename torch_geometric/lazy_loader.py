from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
# python/util/lazy_loader.py
class LazyLoader(ModuleType):
    def __init__(
        self,
        local_name: str,
        parent_module_globals: Dict[str, Any],
        name: str,
    ) -> None:
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self) -> Any:
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        module = self._load()
        return getattr(module, item)

    def __dir__(self) -> List[str]:
        module = self._load()
        return dir(module)
