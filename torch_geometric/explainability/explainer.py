from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.algo.captumexplainer import CaptumExplainer
from torch_geometric.explainability.explanations import Explanation
from torch_geometric.explainability.utils import to_captum


@dataclass
class Threshold:
    type: str
    value: float
    _valid_type = ["soft", "hard", "connected"]

    def __post_init__(self):
        if self.type not in self._valid_type:
            raise ValueError(f'Invalid threshold type {self.type}. '
                             f'Valid types are {self._valid_type}.')
        if self.value < 0 or self.value > 1:
            raise ValueError(f'Invalid threshold value {self.value}. '
                             f'Valid values are in [0, 1].')


@dataclass
class ExplainerConfig:
    explanation_type: str
    mask_type: str
    _valid_explanation_type = ["model", "phenomenon"]
    _valid_mask_type = ["node", "edge", "node_and_edge"]

    def __post_init__(self):
        if self.explanation_type not in self._valid_explanation_type:
            raise ValueError(
                f'Invalid explanation type {self.explanation_type}. '
                f'Valid types are {self._valid_explanation_type}.')
        if self.mask_type not in self._valid_mask_type:
            raise ValueError(f'Invalid mask type {self.mask_type}. '
                             f'Valid types are {self._valid_mask_type}.')


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
            Supported methods are `soft`, `hard`, `connected`.
        loss (torch.nn.Module): the loss function to be used for the
            explanation algorithm.
        model (torch.nn.Module): the GNN module to explain.
        model_return_type (str): the output type of the model.
            Supported outputs are `logits`, `probs`, and `regression`.
        threshold_value (float, optional): the threshold value to be used
            for the `hard` thresholding method. Should be between 0 and 1.
            (default: :obj:`0.5`)
    """
    def __init__(
        self,
        explanation_type: str,
        # provide a factory instead ?
        explanation_algorithm: ExplainerAlgorithm,
        mask_type: str,
        threshold: str,
        loss: torch.nn.Module,
        model: torch.nn.Module,
        model_return_type: str,
        threshold_value: float = 0.5,
    ) -> None:

        super().__init__()

        self.model = model
        # details of the model
        if model_return_type.lower() not in ["logits", "probs", "regression"]:
            raise ValueError(
                "The model return type must be either 'logits', 'probs' or"
                "'regression'.")
        self.model_return_type = model_return_type

        self.explanation_config = ExplainerConfig(
            explanation_type=explanation_type.lower(),
            mask_type=mask_type.lower())

        # details for post-processing the ouput of the explanation algorithm
        self.threshold = Threshold(type=threshold, value=threshold_value)

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
        self.loss = loss
        self.explanation_algorithm.set_objective(self._create_objective())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                y: torch.Tensor, target_index: int = 0,
                **kwargs) -> Explanation:
        """Compute the explanation of the GNN  for the given inputs and target.

        Args:
            x (torch.Tensor): the node features.
            edge_index (torch.Tensor): the edge indices.
            y (torch.Tensor): the target of the GNN.
            target_index (int, optional): the  index of the target to explain.
                If not provided, the explanation is computed for all the first
                index of the target. (default: 0)
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: explanations for the inputs and target.
        """
        target = y

        if self.explanation_config.explanation_type == "model":
            target = self.model(x=x, edge_index=edge_index, **kwargs)

            if self.model_return_type in ["probs", "logits"]:
                target_index = target.argmax()

        if isinstance(self.explanation_algorithm, CaptumExplainer):
            raw_explanation = self._compute_explanation_captum(
                x, edge_index, target, target_index, **kwargs)
        else:
            raw_explanation = self._compute_explanation_pyg(
                x, edge_index, target, target_index, **kwargs)

        return self._post_process_explanation(raw_explanation)

    def _compute_explanation_pyg(self, x: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 target: torch.Tensor, target_index: int,
                                 **kwargs) -> Explanation:
        return self.explanation_algorithm.explain(x=x, edge_index=edge_index,
                                                  target=target,
                                                  model=self.model,
                                                  target_index=target_index,
                                                  **kwargs)

    def _compute_explanation_captum(self, x: torch.Tensor,
                                    edge_index: torch.Tensor,
                                    target: torch.Tensor, target_index: int,
                                    **kwargs) -> Explanation:

        captum_model = to_captum(self.model, self.explanation_config.mask_type,
                                 target_index)

        return self.explanation_algorithm.explain(x=x, edge_index=edge_index,
                                                  target=target,
                                                  model=captum_model,
                                                  target_index=target_index,
                                                  **kwargs)

    def _create_objective(
        self,
    ) -> Callable[[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, int],
                  torch.Tensor]:
        """Creates the objective function for the explanation module depending
        on the loss function and the explanation type.

        If the explanation type is `model`, the objective function is the loss
        function applied on the model output, otherwise it is the loss function
        applied on the true output.
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

        if the thresholding method is `hard`, the mask is thresholded at the
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
        if self.threshold.type == "connected":
            raise NotImplementedError()
        return explanation
