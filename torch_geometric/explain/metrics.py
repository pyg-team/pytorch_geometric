from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explainer, Explanation


class ExplanationMetric(ABC):
    r"""Abstract base class for explanation metrics."""
    def __init__(self, explainer, index) -> None:
        self.explainer = explainer
        self.index = index

    @abstractmethod
    def __call__(self, explainer: Explainer, **kwargs):
        r"""Computes the explanation metric for given explainer and explanation
            Args:
        explainer :obj:`~torch_geometric.explain.Explainer`
            The explainer to evaluate
        """

    def get_inputs(self):
        r"""Obtain inputs all different inputs over which to compute the
        metrics."""

    @abstractmethod
    def compute_metric(self):
        r"""Compute the metric over all inputs."""

    @abstractmethod
    def aggregate(self):
        r"""Aggregate metrics over all inputs"""


class Fidelity(ExplanationMetric):
    r"""Fidelity+/- Explanation Metric as
    defined in https://arxiv.org/abs/2206.09677"""
    def __init__(self) -> None:
        super().__init__()


def fidelity(explainer: Explainer, explanation: Explanation,
             target: Optional[Tensor] = None,
             index: Optional[Union[int, Tensor]] = None,
             output_type: str = 'raw', **kwargs):
    r"""Evaluate the fidelity of Explainer and given
    explanation produced by explainer

    Args:
        explainer :obj:`~torch_geometric.explain.Explainer`
            The explainer to evaluate
        explanation :obj:`~torch_teometric.explain.Explanation`
            The explanation to evaluate
        target (Tensor, optional): The target prediction, if not provided it
            is inferred from obj:`explainer`, defaults to obj:`None`
        index (Union[int, Tensor]): The explanation target index, for node-
            and edge- level task it signifies the nodes/edges explained
            respectively, for graph-level tasks it is assumed to be None,
            defaults to obj:`None`
    """
    metric_dict = {}

    task_level = explainer.model_config.task_level

    if index != explanation.get('index'):
        raise ValueError(f'Index ({index}) does not match '
                         f'explanation.index ({explanation.index}).')

    # get input graphs
    explanation_graph = explanation.get_explanation_subgraph()  # for fid-
    complement_graph = explanation.get_complement_subgraph()  # for fid+

    # get target
    target = explainer.get_target(x=explanation.x,
                                  edge_index=explanation.edge_index,
                                  **kwargs)  # using full explanation

    # get predictions
    explanation_prediction = explainer.get_prediction(
        x=explanation_graph.x, edge_index=explanation_graph.edge_index,
        **kwargs)
    complement_prediction = explainer.get_prediction(
        x=complement_graph.x, edge_index=complement_graph.edge_index, **kwargs)

    # fix logprob to prob
    if output_type == 'prob' and explainer.model.return_type == 'log_probs':
        target = torch.exp(target)
        explanation_prediction = torch.exp(explanation_prediction)
        complement_prediction = torch.exp(complement_prediction)

    # based on task level
    if task_level == 'graph':
        if index is not None:
            ValueError(
                f'Index for graph level task should be None, got (f{index})')
        # evaluate whole entry
        pass
    elif task_level == 'edge':
        # get edge prediction
        pass  # TODO (blaz)
    elif task_level == 'node':
        # get node prediction(s)
        target = target[index]
        explanation_prediction = explanation_prediction[index]
        complement_prediction = complement_prediction[index]
    else:
        raise NotImplementedError

    with torch.no_grad():
        if explainer.model_config.mode == 'regression':
            metric_dict['fidelity-'] = torch.mean(
                torch.abs(target - explanation_prediction))
            metric_dict['fidelity+'] = torch.mean(
                torch.abs(target - complement_prediction))
        elif explainer.model_config.mode == 'classification':
            metric_dict['fidelity-'] = torch.mean(
                target == explanation_prediction)
            metric_dict['fidelity+'] = torch.mean(
                target == complement_prediction)
        else:
            raise NotImplementedError

    return metric_dict
