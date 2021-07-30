from abc import ABC, abstractmethod


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, data):
        pass
