from abc import ABC, abstractmethod


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms."""
    @abstractmethod
    def __call__(self, data):
        pass
