from typing import List, Optional, Tuple, Union

import torch

from torch_geometric.explain.base import ExplainerAlgorithm
from torch_geometric.explain.configuration import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelReturnType,
    ModelTaskLevel,
    Threshold,
    ThresholdType,
)
from torch_geometric.explain.explanations import Explanation


class Explainer:
    """A user configuration class for the instance-level explanation of GNNS.

    Args:
        explanation_algorithm (ExplainerAlgorithm): explanation algorithm
            to be used.
        model (torch.nn.Module): the model to be explained.
        model_return_type (str): denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns
            raw scores) and :obj:`"regression"` (the model returns scalar
            values).(default: :obj:`"log_prob"`)
        task_level (str, optional): type of task the :obj:`model` solves.
            Can be in :obj:`"graph"` (e.g. graph classification/ regression),
            :obj:`"node"` or :obj:`"edge"` (e.g. node/edge classification).
            (default: :obj:`"graph"`)
        explanation_type (str, optional): :obj:`"phenomenon"` (explanation
            of underlying phenomon) or :obj:`"model"` (explanation of model
            prediction behaviour). See  "GraphFramEx: Towards Systematic
            Evaluation of Explainability Methods for Graph Neural Networks"
            <https://arxiv.org/abs/2206.09677> for more details.
            (default: :obj:`"model"`)
        node_mask_type (str, optional): type of node mask to be used.
            Can be in :obj:`"object"` (masking nodes), :obj:`"attributes"`
            (masking the node features), :obj:`"both"` (object and attributes),
            or :obj:`"none"` (no node masking).
        edge_mask_type (str, optional): type of edge mask to be used.
            Same options as :obj:`node_mask_type`.
        threshold (str, optional): type of threshold to apply after the
            explanation algorithm. Valid inputs are :obj:`"none"`,
            :obj:`"hard"`, :obj:`"topk"`, :obj:`"topk_hard"`, and
            :obj:`"connected"`. The thresholding is applied to each mask
            idependently. For the thresholding requirring a count, if the
            :obj:`threshold_value` is bigger than the number of element in the
            mask, it will just return the mask. (default: :obj:`"none"`)

            1. :obj:`"none"`: no thresholding is applied.

            2. :obj:`"hard"`: the mask is thresholded to binary values:
               values above the threshold are set to 1, and values below the
               threshold are set to :obj:`0`.

            3. :obj:`"topk_hard"`: the mask is thresholded to binary
               values: the :obj:`threshold_value` largest values are set
               to :obj:`1`, and the rest are set to :obj:`0`.

            4. :obj:`"topk"`: the mask is thresholded to values between
               :obj:`0` and :obj:`1`: the :obj:`threshold_value` largest
               values are left unchanged, and the rest are set to :obj:`0`.

            5. :obj:`"connected"`: the mask is thresholded to binary values
               such that a connected component of size at least
               :obj:`threshold_value` is kept. The rest is set to :obj:`0`.

        threshold_value (Union[float,int]): Value to use for thresholding.
            If :obj:`threshold` is :obj:`"hard"`, the value should be in
            :obj:`[0,1]`. If :obj:`threshold` is :obj:`"topk"` or
            :obj:`"connected"`, the value should be a positive integer.
            (default: :obj:`1`)

    Raises:
        ValueError: for invalid inputs or if the explainer algorithm does
            not support the given explanation settings
    """
    def __init__(
        self,
        explanation_algorithm: ExplainerAlgorithm,  # TODO: code factory so
        # that user can pass a string and we create the right algorithm with
        # the right parameters if possible
        model: torch.nn.Module,
        model_return_type: Union[str, ModelReturnType],
        task_level: Union[str, ModelTaskLevel] = ModelTaskLevel.graph,
        explanation_type: Union[str, ExplanationType] = ExplanationType.model,
        node_mask_type: Union[str, MaskType] = MaskType.object,
        edge_mask_type: Union[str, MaskType] = MaskType.none,
        threshold: Union[str, ThresholdType] = ThresholdType.none,
        threshold_value: float = 1,
    ) -> None:

        self.model = model
        self.model_config = ModelConfig(return_type=model_return_type,
                                        task_level=task_level)

        self.explanation_config = ExplainerConfig(
            explanation_type=explanation_type, node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type)

        # details for post-processing the output of the explanation algorithm
        self.threshold = Threshold(type=threshold, value=threshold_value)

        # details of the explanation algorithm
        self.explanation_algorithm: ExplainerAlgorithm = explanation_algorithm

        # check that the explanation algorithm supports the
        # desired setup
        if not self.explanation_algorithm.supports(self.explanation_config):
            raise ValueError(
                "The explanation algorithm does not support the configuration."
            )

    def get_prediction(self, x: torch.Tensor, edge_index: torch.Tensor,
                       batch: Optional[torch.Tensor] = None,
                       **kwargs) -> torch.Tensor:
        r"""Returns the prediction of the model on the input graph.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor, optional): the batch vector.
            **kwargs: Additional arguments to be passed to the model.
        """
        with torch.no_grad():
            return self.model(x=x, edge_index=edge_index, batch=batch,
                              **kwargs)

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor,
                 target: Optional[torch.Tensor] = None,
                 target_index: Union[int, Tuple[int, ...], torch.Tensor,
                                     List[Tuple[int, ...]], List[int]] = 0,
                 batch: Optional[torch.Tensor] = None,
                 **kwargs) -> Explanation:
        """Compute the explanation of the GNN  for the given inputs and target.

        Args:
            x (torch.Tensor): the node features.
            edge_index (torch.Tensor): the edge indices.
            target (torch.Tensor, optional): the target of the GNN. If the
                explanation type is :obj:`"phenomenon"`, the target has to be
                provided. If the explanation type is :obj:`"model"`, the target
                will be replaced by the model output and can be :obj:`None`.
                (default: :obj:`None`)
            target_index (TargetIndex): Output indices to explain.
                If not provided, the explanation is computed for the first
                index of the target. (default: :obj:`0`)

                For general 1D outputs, targets can be either:

                - a single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                  the number of examples (i.e number of unique values in
                  the batch vector). Each integer is applied as the
                  target for the corresponding element of the batch.

                For outputs with > 1 dimension, targets can be either:

                - a single tuple, which contains (:obj:`target.dim()`)
                  elements. This target index is applied for all
                  elements of the batch.

                - a list of tuples with length equal to the number of
                  examples in inputs, and each tuple containing
                  (:obj:`target.dim()`) elements. Each tuple is applied
                  as the target for the corresponding element of the
                  batch.

            batch (torch.Tensor, optional): the batch vector.  If not provided,
                suppose only one input.(default::obj:`None`)
            **kwargs: additional arguments to pass to the GNN.

        Raises:
            ValueError: if the explanation type is :obj:`"phenomenon"` and the
                target is not provided.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.explanation_config.explanation_type.value == "phenomenon":
            if target is None:
                raise ValueError(
                    "The target has to be provided for the explanation type "
                    f"'{self.explanation_config.explanation_type}'")
        else:
            target = self.get_prediction(x=x, edge_index=edge_index,
                                         batch=batch, **kwargs)

        # TODO: add check that target_index is valid based on value of batch

        raw_explanation = self.explanation_algorithm.forward(
            x=x, edge_index=edge_index, model=self.model, target=target,
            target_index=target_index, batch=batch,
            task_level=self.model_config.task_level,
            return_type=self.model_config.return_type,
            node_mask_type=self.explanation_config.node_mask_type,
            edge_mask_type=self.explanation_config.edge_mask_type, **kwargs)

        return self._post_process_explanation(raw_explanation)

    def _post_process_explanation(self,
                                  explanation: Explanation) -> Explanation:
        """Post-process the explanation mask according to the thresholding
        method and the user configuration.

        Args:
            explanation (Explanation): the explanation mask to post-process.
        """
        explanation = self._threshold(explanation)
        return explanation

    def _threshold(self, explanation: Explanation) -> Explanation:
        """Threshold the explanation mask according to the thresholding method.

            Args:
                explanation (Explanation): explanation to threshold.

            Raises:
                NotImplementedError: if the thresholding method is connected.
        """

        # avoid modification of the original explanation
        explanation = explanation.clone()

        # get the available mask
        available_mask_keys = explanation.available_explanations
        available_mask = [explanation[key] for key in available_mask_keys]

        if self.threshold.type.value == "hard":
            available_mask = [(mask > self.threshold.value).float()
                              for mask in available_mask]

        if self.threshold.type.value in ["topk", "topk_hard"]:
            updated_masks = []
            for mask in available_mask:
                if self.threshold.value >= mask.numel():
                    updated_mask = mask
                else:
                    topk_values, indices = torch.topk(input=mask.flatten(),
                                                      k=self.threshold.value)
                    updated_mask = torch.zeros_like(mask.flatten()).scatter_(
                        0, indices, topk_values)

                if self.threshold.type.value == "topk_hard":
                    updated_mask = (updated_mask > 0).float()
                updated_masks.append(updated_mask.reshape(mask.shape))
            available_mask = updated_masks

        if self.threshold.type.value == "connected":
            raise NotImplementedError()

        # update the explanation with the thresholded masks
        for key, mask in zip(available_mask_keys, available_mask):
            explanation[key] = mask
        return explanation
