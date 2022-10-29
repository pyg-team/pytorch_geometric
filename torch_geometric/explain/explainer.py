from dataclasses import dataclass
from inspect import getfullargspec
from typing import Optional, Union

import torch

from torch_geometric.data import Data
from torch_geometric.explain.base import ExplainerAlgorithm
from torch_geometric.explain.explanations import Explanation


@dataclass
class Threshold:
    """Class to store and validate threshold parameters."""
    type: str
    value: Union[float, int]
    _valid_type = ["none", "hard", "topk", "connected"]

    def __post_init__(self):
        if self.type not in self._valid_type:
            raise ValueError(f'Invalid threshold type {self.type}. '
                             f'Valid types are {self._valid_type}.')
        if not isinstance(self.value, (float, int)):
            raise ValueError(f'Invalid threshold value {self.value}. '
                             f'Valid values are float or int.')

        if self.type == "hard":
            if self.value < 0 or self.value > 1:
                raise ValueError(f'Invalid threshold value {self.value}. '
                                 f'Valid values are in [0, 1].')
        if self.type in ["topk", "connected"]:
            if self.value <= 0 or not isinstance(self.value, int):
                raise ValueError(f'Invalid threshold value {self.value}. '
                                 f'Valid values are positif integers.')


@dataclass
class ExplainerConfig:
    """Class to store and validate high level explanation parameters."""
    explanation_type: str
    mask_type: str
    _valid_explanation_type = ["model", "phenomenon"]
    _valid_mask_type = [
        "node", "edge", "node_and_edge", "layers", "node_feat", "edge_feat",
        "node_feat_and_edge_feat", "node_feat_and_edge", "node_and_edge_feat"
    ]

    def __post_init__(self):
        if self.explanation_type not in self._valid_explanation_type:
            raise ValueError(
                f'Invalid explanation type {self.explanation_type}. '
                f'Valid types are {self._valid_explanation_type}.')
        if self.mask_type not in self._valid_mask_type:
            raise ValueError(f'Invalid mask type {self.mask_type}. '
                             f'Valid types are {self._valid_mask_type}.')


@dataclass
class ModelConfig:
    model: torch.nn.Module
    return_type: str
    task_level: str = "graph"
    mode: str = "classification"
    _valid_model_return_type = ["logits", "probs", "raw", "regression"]
    _valid_model_mode = ["classification", "regression"]
    _valid_task = ["graph", "node", "edge"]

    def __post_init__(self):

        if self.return_type not in self._valid_model_return_type:
            raise ValueError(
                f'Invalid model return type {self.return_type}. '
                f'Valid types are {self._valid_model_return_type}.')
        if self.mode not in self._valid_model_mode:
            raise ValueError(f'Invalid model mode {self.mode}. '
                             f'Valid modes are {self._valid_model_mode}.')

        if self.task_level not in self._valid_task:
            raise ValueError(f'Invalid task {self.task_level}. '
                             f'Valid tasks are {self._valid_task}.')

        if Data not in getfullargspec(self.model.forward).annotations.values():
            message = "The model does not accept a `Data` object as input."
            message += "Consider using `explain.utils.Interface` to wrap the"
            message += "forward function of your model."
            message += "Check the documentation for more details."

            raise ValueError(message)

        # update model mode based on return type
        if self.return_type == "regression":
            self.mode = "regression"


class Explainer(torch.nn.Module):
    r"""A user configuration class for the instance-level explanation of GNNS.

        Args:
            explanation_algorithm (ExplainerAlgorithm): explanation algorithm
                to be used. Should accept a `Data` object as input.
                (see py:class:Interface for wrapping a model if needed).
            model_return_type (str): denotes the type of output from
                :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
                returns the logarithm of probabilities), :obj:`"prob"` (the
                model returns probabilities), :obj:`"raw"` (the model returns
                raw scores) and :obj:`"regression"` (the model returns scalars)
                (default: :obj:`"log_prob"`)
            task_level (str, optional): type of task the :obj:`model` solves.
                Can be in :obj:"graph" (e.g. graph classification/ regression),
                :obj:"node" or :obj:"edge" (e.g. node/edge classificaiton).
                (default: :obj:`"graph"`)
            explanation_type (str, optional): :obj:`"phenomenon"` (explanation
                of underlying phenomon) or :obj:`"model"` (explanation of model
                prediction behaviour). See  "GraphFramEx: Towards Systematic
                Evaluation of Explainability Methods for Graph Neural Networks"
                <https://arxiv.org/abs/2206.09677> for more details.
                (default: :obj:`"model"`)
            mask_type (str, optional): Type of mask wanted. The masks are
                between 0 and 1, and can be on the features of the nodes/edges,
                or on the nodes/edges themselves. Valid inputs are
                :obj:`"node"`, :obj:`"edge"`, :obj:`"node_and_edge"`,
                :obj:`"layers"`, :obj:`"node_feat"`, :obj:`"edge_feat"`,
                :obj:`"node_feat_and_edge_feat"`, :obj:`"node_feat_and_edge"`,
                :obj:`"node_and_edge_feat"`.

            threshold (str, optional): type of threshold to apply after the
                explanation algorithm. Valid inputs are :obj:`"nonde"`,
                :obj:`"hard"`, :obj:`"topk"` and :obj:`"connected"`.
                The thresholding is applied to each mask idependently.

                1. :obj:`"none"`: no thresholding is applied.

                2. :obj:`"hard"`: the mask is thresholded to binary values:
                values above the threshold are set to 1, and values below the
                threshold are set to 0.

                3. :obj:`"topk"`: the mask is thresholded to binary values:
                    the :obj:`threshold_value` largest values are set to 1,
                    and the rest are set to 0.

                4. :obj:`"connected"`: the mask is thresholded to binary values
                    such that a connected component of size at least
                    :obj:`threshold_value` is kept. The rest is set to 0.
                    NotImplemnted for now.

            threshold_value (Union[float,int]): Value to use for thresholding.
                If :obj:`threshold` is :obj:`"hard"`, the value should be in
                [0,1]. If :obj:`threshold` is :obj:`"topk"` or
                :obj:`"connected"`, the value should be a positive integer.
                (default: :obj:`1`)

        Raises:
            ValueError: for invalid inputs or if the explainer algorithm does
                not support the given explanation settings

        . note::
        If the model you are trying to explain does not take a `Data` object as
        input, you can use the `Interface` class to help you create a wrapper.

    .. code-block:: python
        class GCNCcompatible(GCN):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.interface = Interface(graph_to_inputs=lambda x: {
                    "x": x.x,
                    "edge_index": x.edge_index
                })

            def forward(self, g, **kwargs):
                return super().forward(
                    **self.interface.graph_to_inputs(g, **kwargs)
                )
    """
    def __init__(
        self,
        explanation_algorithm: ExplainerAlgorithm,  # TODO: code factory so
        # that user can pass a string and we create the right algorithm with
        # the right parameters if possible
        model: torch.nn.Module,
        model_return_type: str,
        task_level: str = "graph",
        explanation_type: str = "model",
        mask_type: str = "node",
        threshold: str = "none",
        threshold_value: float = 1,
    ) -> None:

        super().__init__()

        self.model_config = ModelConfig(model=model,
                                        return_type=model_return_type.lower(),
                                        task_level=task_level.lower())

        self.explanation_config = ExplainerConfig(
            explanation_type=explanation_type.lower(),
            mask_type=mask_type.lower())

        # details for post-processing the ouput of the explanation algorithm
        self.threshold = Threshold(type=threshold.lower(),
                                   value=threshold_value)

        # details of the explanation algorithm
        self.explanation_algorithm = explanation_algorithm

        # check that the explanation algorithm supports the
        # desired setup
        if not self.explanation_algorithm.supports(
                self.explanation_config.explanation_type,
                self.explanation_config.mask_type):
            raise ValueError(
                "The explanation algorithm does not support the configuration."
            )

    def get_prediction(self, g: Data, batch=None, **kwargs) -> torch.Tensor:
        r"""Returns the prediction of the model on the input graph.

        Args:
            g (torch_geometric.data.Data): the input graph.
            batch (torch.Tensor, optional): the batch vector.
                (default: :obj:`None`)
        """
        with torch.no_grad():
            return self.model_config.model(g, **dict({
                "batch": batch,
                **kwargs
            }))

    def forward(self, g: Data, target: Optional[torch.Tensor] = None,
                target_index: int = 0, batch: Optional[torch.Tensor] = None,
                **kwargs) -> Explanation:
        """Compute the explanation of the GNN  for the given inputs and target.



        Args:
            g (Data): the input graph.
            target (torch.Tensor, optional): the target of the GNN. If the
                explanation type is :obj:`"phenomenon"`, the target has to be
                provided. If the explanation type is :obj:`"model"`, the target
                will be replaced by the model output and can be :obj:`None`.
                (default: :obj:`None`)
            target_index (int): the  index of the target to explain. If not
                provided, the explanation is computed for the first index of
                the target. (default: :obj:`0`)
            batch (torch.Tensor, optional): the batch vector. (default:
                :obj:`None`)
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: explanations for the inputs and target.

        Raises:
            ValueError: if the explanation type is :obj:`"phenomenon"` and the
                target is not provided.
        """
        if self.explanation_config.explanation_type == "phenomenon":
            if target is None:
                raise ValueError(
                    "The target has to be provided for the explanation type "
                    f"'{self.explanation_config.explanation_type}'")
        else:
            target = self.get_prediction(g, batch, **kwargs)

        raw_explanation = self._compute_explanation(
            g,
            target,  # type: ignore
            target_index,
            batch,
            **kwargs)
        return self._post_process_explanation(raw_explanation)

    def _compute_explanation(self, g: Data, target: torch.Tensor,
                             target_index: int,
                             batch: Optional[torch.Tensor] = None,
                             **kwargs) -> Explanation:

        return self.explanation_algorithm.explain(
            g=g, model=self.model_config.model, target=target,
            target_index=target_index, batch=batch,
            task_level=self.model_config.task_level,
            return_type=self.model_config.return_type, **kwargs)

    def _post_process_explanation(self,
                                  explanation: Explanation) -> Explanation:
        """Post-process the explanation mask according to the thresholding
        method and the user configuration.

        Args:
            explanation (Explanation): the explanation mask to post-process.

        Returns:
            Explanation: the post-processed explanation mask.
        """
        explanation = self._threshold(explanation)
        return explanation

    def _threshold(self, explanation: Explanation) -> Explanation:
        """Threshold the explanation mask according to the thresholding method.

            Args:
                explanation (Explanation): explanation to threshold.

            Raises:
                NotImplementedError: if the thresholding method is connected.

            Returns:
                Explanation: thresholded explanation.
        """
        # retrieve masks that are not None
        available_mask_keys = explanation.available_explanations
        available_mask = [explanation[key] for key in available_mask_keys]

        if self.threshold.type == "hard":
            available_mask = [(mask > self.threshold.value).float()
                              for mask in available_mask]
        if self.threshold.type == "topk":
            raise NotImplementedError()
        if self.threshold.type == "connected":
            raise NotImplementedError()

        # update the explanation with the thresholded masks
        for key, mask in zip(available_mask_keys, available_mask):
            explanation[key] = mask
        return explanation
