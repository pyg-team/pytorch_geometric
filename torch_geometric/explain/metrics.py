from abc import abstractmethod
from typing import Optional

from torch_geometric.explain import Explainer, Explanation


class ExplanationMetric():
    r"""
    Abstract base class for explanation metrics.
    """
    @abstractmethod
    def __call__(self, explainer: Explainer,
                 explanation: Optional[Explanation], **kwargs):
        pass


def _fidelity():
    pass
