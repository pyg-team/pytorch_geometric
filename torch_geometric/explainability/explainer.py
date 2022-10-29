from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from torch_geometric.data import Data
from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.algo.captumexplainer import CaptumExplainer
from torch_geometric.explainability.algo.utils import to_captum
from torch_geometric.explainability.explanations import Explanation
from torch_geometric.explainability.utils import Interface

# TODO: add constraints as arguments of the dataclasses


@dataclass
class Threshold:
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
        if self.type == "topk" or self.type == "connected":
            if self.value < 0 or not isinstance(self.value, int):
                raise ValueError(f'Invalid threshold value {self.value}. '
                                 f'Valid values are positif integers.')
        if self.type == "connected":
            raise NotImplementedError()


@dataclass
class ExplainerConfig:
    explanation_type: str
    mask_type: str
    task: str = "graph_level"
    _valid_explanation_type = ["model", "phenomenon"]
    _valid_mask_type = [
        "node", "edge", "node_and_edge", "layers", "node_feat", "edge_feat",
        "node_and_edge_feat", "node_feat_and_edge", "edge_feat_and_node"
    ]
    _valid_task = ["graph_level", "node_level"]

    def __post_init__(self):
        if self.explanation_type not in self._valid_explanation_type:
            raise ValueError(
                f'Invalid explanation type {self.explanation_type}. '
                f'Valid types are {self._valid_explanation_type}.')
        if self.mask_type not in self._valid_mask_type:
            raise ValueError(f'Invalid mask type {self.mask_type}. '
                             f'Valid types are {self._valid_mask_type}.')
        if self.mask_type == "layers":
            raise NotImplementedError()
        if self.task not in self._valid_task:
            raise ValueError(f'Invalid task {self.task}. '
                             f'Valid tasks are {self._valid_task}.')


@dataclass
class ModelConfig:
    model: torch.nn.Module
    return_type: str
    interface: Interface = Interface()
    _valid_model_return_type = ["logits", "probs", "regression"]

    def __post_init__(self):
        if self.return_type not in self._valid_model_return_type:
            raise ValueError(
                f'Invalid model return type {self.return_type}. '
                f'Valid types are {self._valid_model_return_type}.')


class Explainer(torch.nn.Module):
    r"""A user configuration class for the instance-level explanation
    of Graph Neural Networks, *e.g.* :class:`~torch_geometric.nn.GCN`.

    Args:
        explanation_type (str): the type of explanation to be computed.
            Supported types are `model` and `phenomenon`.
        explanation_algorithm (ExplainerAlgorithm): the explanation
            algorithm to be used.
        mask_type (str): the type of mask to be used. Supported types are
            `node`, `edge`, `global`. `global` returns masks for both nodes
            and edges.
        threshold (str): post-processing thresholding method on the mask.
            Supported methods are `none`, `soft`, `topk`, `connected`.
        loss (torch.nn.Module): the loss function to be used for the
            explanation algorithm.
        model (torch.nn.Module): the GNN module to explain.
        model_return_type (str): the output type of the model.
            Supported outputs are `logits`, `probs`, and `regression`.
        threshold_value (float, optional): the threshold value to be used
            for the `hard` thresholding method. Should be between 0 and 1.
            (default: :obj:`0.5`)

    .. note::
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
        explanation_type: str,
        explanation_algorithm: ExplainerAlgorithm,  # TODO: code factory so
        # that user can pass a string and we create the right algorithm with
        # the right parameters if possible
        mask_type: str,
        threshold: str,
        model: torch.nn.Module,
        model_return_type: str,
        loss: Optional[torch.nn.Module] = None,
        threshold_value: float = 0.5,
        task_level: str = "graph_level",
    ) -> None:

        super().__init__()

        self.model_config = ModelConfig(model=model,
                                        return_type=model_return_type.lower())

        self.explanation_config = ExplainerConfig(
            explanation_type=explanation_type.lower(),
            mask_type=mask_type.lower(), task=task_level.lower())

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

        # update the loss function of the explanation algorithm based on setup
        if loss is not None and self.explanation_algorithm.accept_new_loss:
            self.loss = loss
        else:
            self.loss = self.explanation_algorithm.loss

        self.explanation_algorithm.set_objective(self._create_objective())

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

    def forward(self, g: Data, y: torch.Tensor,
                target_index: Optional[int] = None, batch: torch.Tensor = None,
                **kwargs) -> Explanation:
        """Compute the explanation of the GNN  for the given inputs and target.

        Args:
            g (Data): the input graph.
            y (torch.Tensor): the target of the GNN.
            target_index (int, optional): the  index of the target to explain.
                If not provided, the explanation is computed for the first
                index of the target. (default: :obj:`None`)
            batch (torch.Tensor, optional): the batch vector. (default: None)
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: explanations for the inputs and target.
        """
        target = y

        if self.explanation_config.explanation_type == "model":
            target = self.get_prediction(g, batch, **kwargs)

            if self.model_config.return_type in ["probs", "logits"]:
                target_index = target.argmax()

        raw_explanation = self._compute_explanation(g, target, target_index,
                                                    batch, **kwargs)
        return self._post_process_explanation(raw_explanation)

    def _compute_explanation(self, g: Data, target: torch.Tensor,
                             target_index: int, batch: torch.Tensor,
                             **kwargs) -> Explanation:

        if isinstance(self.explanation_algorithm, CaptumExplainer):
            captum_model = to_captum(self.model_config.model,
                                     self.explanation_config.mask_type,
                                     target_index)
            return self.explanation_algorithm.explain(
                g=g, model=captum_model, target=target,
                target_index=target_index, batch=batch, **kwargs)

        return self.explanation_algorithm.explain(
            g=g, model=self.model_config.model, target=target,
            target_index=target_index, batch=batch, **kwargs)

    def _create_objective(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int],
                  torch.Tensor]:
        """Creates the objective function for the explanation module depending
        on the loss function and the explanation type.

        If the explanation type is `model`, the objective function is the loss
        function applied on the model output, otherwise it is the loss function
        applied on the true output.

        The returned function takes as input the explanation_output, the model
        output, the target and the target index and returns the loss value.
        The explanation output can be, for example, the output of the model
        after applying a mask on some of its inputs.
        """
        if self.explanation_config.explanation_type == "model":
            return lambda x, y, z, i: self.loss(x[i], y[i])

        return lambda x, y, z, i: self.loss(x[i], z[i])

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

        if the thresholding method is `topk`, the mask is thresholded at the
        threshold value.
        if the thresholding method is `soft`, the mask is returned as is.

        Args:
            explanation (Explanation): explanation to threshold.

        Raises:
            NotImplementedError: if the thresholding method is connected.

        Returns:
            Explanation: threhsolded explanation.
        """
        if self.threshold.type == "hard":
            explanation.threshold(self.threshold.value)
        if self.threshold.type == "topk":
            explanation.threshold_topk(self.threshold.value)
        if self.threshold.type == "connected":
            raise NotImplementedError()
        return explanation
