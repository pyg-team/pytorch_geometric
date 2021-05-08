from typing import Any

import copy


class Storage(dict):
    # This class extends the Python dictionary class by the following:
    # 1. It allows attribute assignment, e.g., `storage.x = ...` rather than
    #    `storage['x'] = ...`.
    # 2. It hides local keys/attributes to the user, e.g., `storage._x`.
    def __getattr__(self, key: str) -> Any:
        return self.get(key, None)

    def __setattr__(self, key: str, item: Any):
        if item is None and key in self:
            del self[key]
        elif item is not None:
            self[key] = item

    def __delattr__(self, key: str):
        if key in self:
            del self[key]

    def __copy__(self):
        out = self.__class__()
        out.update(self)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, item in self.items():
            out[key] = copy.deepcopy(item, memo)
        return out
