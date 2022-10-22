from typing import Callable, Tuple

import torch

from torch_geometric.explainability.algo.base import ExplainerAlgorithm
from torch_geometric.explainability.algo.captumexplainer import CaptumExplainer
from torch_geometric.explainability.explanations import Explanation
from torch_geometric.explainability.utils import to_captum


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
    ) -> None:

        super().__init__()

        self.model = model

        # high level type of explanations
        if explanation_type.lower() not in ["model", "phenomenon"]:
            raise ValueError(
                "The explanation type must be either 'model' or 'phenomenon'.")
        self.explanation_type = explanation_type

        # details of desired output
        if mask_type.lower() not in ["node", "edge", "node_and_edge"]:
            raise ValueError(
                "The mask type must be in 'node', 'edge' or 'node_and_edge'.")
        self.mask_type = mask_type

        # details for post-processing the ouput of the explanation algorithm
        if threshold.lower() not in ["soft", "hard", "connected"]:
            raise ValueError(
                "The thresholding method must be either 'soft', 'hard' or"
                "'connected'.")
        self.threshold = threshold

        # details of the model
        if model_return_type.lower() not in ["logits", "probs", "regression"]:
            raise ValueError(
                "The model return type must be either 'logits', 'probs' or"
                "'regression'.")
        self.model_return_type = model_return_type

        # details of the explanation algorithm
        self.explanation_algorithm = explanation_algorithm
        if isinstance(self.explanation_algorithm, CaptumExplainer):
            self.backend = "captum"
        else:
            self.backend = "pyg"

        # check that the explanation algorithm supports the
        # desired setup
        if not self.explanation_algorithm.supports(self.explanation_type,
                                                   self.mask_type):
            raise ValueError(
                "The explanation algorithm does not support the configuration."
            )

        # update the loss function of the explanation algorithm based on setup
        self.loss = loss
        self.explanation_algorithm.set_objective(self._create_objective())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                y: torch.Tensor, **kwargs) -> Explanation:
        """Compute the explanation of the GNN  for the given inputs and target.

        Args:
            x (torch.Tensor): the node features.
            edge_index (torch.Tensor): the edge indices.
            y (torch.Tensor): the target of the GNN.
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: explanations for the inputs and target.
        """
        target = y
        if self.explanation_type == "model":
            target = self.model(x=x, edge_index=edge_index, **kwargs)

        if self.model_return_type in ["probs", "logits"]:
            target = target.argmax()

        if self.backend == "pyg":
            raw_explanation = self._compute_explanation_pyg(
                x, edge_index, target, **kwargs)
        elif self.backend == "captum":
            raw_explanation = self._compute_explanation_captum(
                x, edge_index, target, **kwargs)
        return self._post_process_explanation(raw_explanation)

    def _compute_explanation_pyg(self, x: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 target: torch.Tensor,
                                 **kwargs) -> Explanation:
        return self.explanation_algorithm.explain(x=x, edge_index=edge_index,
                                                  target=target,
                                                  model=self.model, **kwargs)

    def _compute_explanation_captum(self, x: torch.Tensor,
                                    edge_index: torch.Tensor,
                                    target: torch.Tensor,
                                    **kwargs) -> Explanation:

        captum_model = to_captum(self.model, self.mask_type, 0)

        return self.explanation_algorithm.explain(x=x, edge_index=edge_index,
                                                  target=target,
                                                  model=captum_model, **kwargs)

    def _create_objective(
        self,
    ) -> Callable[[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
                  torch.Tensor]:
        """Creates the objective function for the explanation module depending
        on the loss function and the explanation type.

        If the explanation type is `model`, the objective function is the loss
        function applied on the model output, otherwise it is the loss function
        applied on the true output.
        """
        if self.explanation_type == "model":
            return lambda x, y, z: self.loss(x, y)
        else:
            return lambda x, y, z: self.loss(x, z)

    def _post_process_explanation(self,
                                  explanation: Explanation) -> Explanation:
        """Post-process the explanation mask according to the thresholding
        method and the user configuration.

        Args:
            explanation (Explanation): the explanation mask to post-process.

        Returns:
            Explanation: the post-processed explanation mask.
        """
        raise NotImplementedError()
