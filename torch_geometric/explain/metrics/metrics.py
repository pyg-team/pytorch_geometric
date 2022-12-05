from abc import ABC, abstractmethod
from typing import Optional

from torch_geometric.explain import Explainer, Explanation


class ExplanationMetric(ABC):
    r"""Abstract base class for explanation metrics."""
    @abstractmethod
    def __call__(self, explainer: Explainer,
                 explanation: Optional[Explanation], **kwargs):
        r"""Computes the explanation metric for given explainer and explanation
            Args:
        explainer :obj:`~torch_geometric.explain.Explainer`
            The explainer to evaluate
        explanation :obj:`~torch_teometric.explain.Explanation`
            The explanation to evaluate
        """

    @abstractmethod
    def get_inputs(self):
        r"""Obtain inputs all different inputs over which to compute the
        metrics."""

    @abstractmethod
    def compute_metric(self):
        r"""Compute the metric over all inputs."""

    @abstractmethod
    def aggregate(self):
        r"""Aggregate metrics over all inputs"""
