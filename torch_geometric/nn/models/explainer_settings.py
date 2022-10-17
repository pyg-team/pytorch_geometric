from typing import Tuple

import torch

from torch_geometric.nn.models.explainer_algorithms import ExplainerAlgorithm


class ExplainerSetter(torch.nn.Module):
    r"""A user configuration class for the instance level-explanation
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
        if explanation_type.lower() not in ["model", "phenomenon"]:
            raise ValueError(
                "The explanation type must be either 'model' or 'phenomenon'.")
        self.explanation_type = explanation_type

        # TODO: how to handle the case where the attribution are on layers?
        # should mask differentiate between node level vs node feature level?
        #  same for edge ? or should we just add a flag to aggregate the mask
        # for the feature to the node/edge level ?
        if mask_type.lower() not in ["node", "edge", "global"]:
            raise ValueError(
                "The mask type must be either 'node', 'edge' or 'global'.")
        self.mask_type = mask_type

        if threshold.lower() not in ["soft", "hard", "connected"]:
            raise ValueError(
                "The thresholding method must be either 'soft', 'hard' or"
                "'connected'.")
        self.threshold = threshold

        if model_return_type.lower() not in ["logits", "probs", "regression"]:
            raise ValueError(
                "The model return type must be either 'logits', 'probs' or"
                "'regression'.")
        self.model_return_type = model_return_type

        self.explanation_algorithm = explanation_algorithm
        if not self.explanation_algorithm.support(self.explanation_type,
                                                  self.mask_type):
            raise ValueError(
                "The explanation algorithm does not support the configuration."
            )
        # how to check the case where the explaination algorithm does not
        # accept a loss ? raise error in the explanation algorithm ?
        # should this method instantiate the explaination algorithm ?
        self.loss = loss
        self.explanation_algorithm._set_loss(self._create_objective())

        self.model = model

    # TODO: have to add target index and check that the user did not provide
    # model level explanation
    def forward(self, inputs: Tuple[torch.Tensor, ...],
                target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute the explanation of the GNN  for the given inputs and target.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            target (torch.Tensor): the target of the GNN.

        Returns:
            Tuple[torch.Tensor, ...]: explanations for the inputs and target.
            #TODO: describe all possible outcomes depending on user input ?
        """
        explanation_masks = self._compute_explanation(inputs, target)
        return self._post_process_explanation(explanation_masks)

    def _create_objective(self):
        """Creates the objective function for the explanation module depending
        on the loss function and the explanation type.

        If the explanation type is `model`, the objective function is the loss
        function applied on the model output, otherwise it is the loss function
        applied on the true output.
        """
        raise NotImplementedError()

    def _compute_explanation(self, inputs: Tuple[torch.Tensor, ...],
                             target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute the explanation mask according to the explanation algorithm
        and the user configuration.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            target (torch.Tensor): the target of the GNN.

        Shape:
            - Output: :obj:`Tuple[torch.Tensor, ...]` a list of masks depending
            on the mask type.
            If mask type is `node`, the output is a list of node masks,
            if mask type is `edge`, the output is a list of edge masks,
            if mask type is `global`, the output is a list of masks for nodes
            and edges.
        """

        return self.explanation_algorithm.explain(inputs, target, self.model)

    def _post_process_explanation(
        self, explanation_masks: Tuple[torch.Tensor,
                                       ...]) -> Tuple[torch.Tensor, ...]:
        """Post-process the explanation mask according to the thresholding
        method and the user configuration."""
        raise NotImplementedError()
