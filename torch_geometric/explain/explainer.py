import copy
import warnings
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import ExplainerAlgorithm, Explanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ThresholdConfig,
    ThresholdType,
)


class Explainer:
    r"""An explainer class for instance-level explanations of Graph Neural
    Networks.

    Args:
        model (torch.nn.Module): The model to explain.
        algorithm (ExplainerAlgorithm): The explanation algorithm.
        explanation_type (ExplanationType or str): The type of explanation to
            compute. The possible values are:

                - :obj:`"model"`: Explains the model prediction.

                - :obj:`"phenomenon"`: Explains the phenomenon that the model
                  is trying to predict.

            In practice, this means that the explanation algorithm will either
            compute their losses with respect to the model output
            (:obj:`"model"`) or the target output (:obj:`"phenomenon"`).
        model_config (ModelConfig): The model configuration.
            See :class:`~torch_geometric.explain.config.ModelConfig` for
            available options. (default: :obj:`None`)
        node_mask_type (MaskType or str, optional): The type of mask to apply
            on nodes. The possible values are (default: :obj:`None`):

                - :obj:`None`: Will not apply any mask on nodes.

                - :obj:`"object"`: Will mask each node.

                - :obj:`"common_attributes"`: Will mask each feature.

                - :obj:`"attributes"`: Will mask each feature across all nodes.

        edge_mask_type (MaskType or str, optional): The type of mask to apply
            on edges. Has the sample possible values as :obj:`node_mask_type`.
            (default: :obj:`None`)
        threshold_config (ThresholdConfig, optional): The threshold
            configuration.
            See :class:`~torch_geometric.explain.config.ThresholdConfig` for
            available options. (default: :obj:`None`)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        algorithm: ExplainerAlgorithm,
        explanation_type: Union[ExplanationType, str],
        model_config: Union[ModelConfig, Dict[str, Any]],
        node_mask_type: Optional[Union[MaskType, str]] = None,
        edge_mask_type: Optional[Union[MaskType, str]] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        explainer_config = ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        )

        self.model = model
        self.algorithm = algorithm
        self.explanation_type = explainer_config.explanation_type
        self.model_config = ModelConfig.cast(model_config)
        self.node_mask_type = explainer_config.node_mask_type
        self.edge_mask_type = explainer_config.edge_mask_type
        self.threshold_config = ThresholdConfig.cast(threshold_config)

        self.algorithm.connect(explainer_config, self.model_config)

    @torch.no_grad()
    def get_prediction(self, *args, **kwargs) -> torch.Tensor:
        r"""Returns the prediction of the model on the input graph.

        If the model mode is :obj:`"regression"`, the prediction is returned as
        a scalar value.
        If the model mode :obj:`"classification"`, the prediction is returned
        as the predicted class label.

        Args:
            *args: Arguments passed to the model.
            **kwargs (optional): Additional keyword arguments passed to the
                model.
        """
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            out = self.model(*args, **kwargs)
        if self.model_config.mode == ModelMode.classification:
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Optional[Tensor] = None,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        r"""Computes the explanation of the GNN for the given inputs and
        target.

        .. note::

            If you get an error message like "Trying to backward through the
            graph a second time", make sure that the target you provided
            was computed with :meth:`torch.no_grad`.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The input edge indices.
            target (torch.Tensor): The target of the model.
                If the explanation type is :obj:`"phenomenon"`, the target has
                to be provided.
                If the explanation type is :obj:`"model"`, the target should be
                set to :obj:`None` and will get automatically inferred.
                (default: :obj:`None`)
            index (Union[int, Tensor], optional): The index of the model
                output to explain. Can be a single index or a tensor of
                indices. (default: :obj:`None`)
            target_index (int, optional): The index of the model outputs to
                reference in case the model returns a list of tensors, *e.g.*,
                in a multi-task learning scenario. Should be kept to
                :obj:`None` in case the model only returns a single output
                tensor. (default: :obj:`None`)
            **kwargs: additional arguments to pass to the GNN.
        """
        # Choose the `target` depending on the explanation type:
        if self.explanation_type == ExplanationType.phenomenon:
            if target is None:
                raise ValueError(
                    f"The 'target' has to be provided for the explanation "
                    f"type '{self.explanation_type.value}'")
        elif self.explanation_type == ExplanationType.model:
            if target is not None:
                warnings.warn(
                    f"The 'target' should not be provided for the explanation "
                    f"type '{self.explanation_type.value}'")
            target = self.get_prediction(x=x, edge_index=edge_index, **kwargs)

        training = self.model.training
        self.model.eval()

        explanation = self.algorithm(
            self.model,
            x,
            edge_index,
            target=target,
            index=index,
            target_index=target_index,
            **kwargs,
        )

        self.model.train(training)

        return self._post_process(explanation)

    def _post_process(self, explanation: Explanation) -> Explanation:
        R"""Post-processes the explanation mask according to the thresholding
        method and the user configuration.

        Args:
            explanation (Explanation): The explanation mask to post-process.
        """
        explanation = self._threshold(explanation)
        return explanation

    def _threshold(self, explanation: Explanation) -> Explanation:
        """Threshold the explanation mask according to the thresholding method.

        Args:
            explanation (Explanation): The explanation to threshold.
        """
        if self.threshold_config is None:
            return explanation

        # Avoid modification of the original explanation:
        explanation = copy.copy(explanation)

        mask_dict = {  # Get the available masks:
            key: explanation[key]
            for key in explanation.available_explanations
        }

        if self.threshold_config.type == ThresholdType.hard:
            mask_dict = {
                key: (mask > self.threshold_config.value).float()
                for key, mask in mask_dict.items()
            }

        elif self.threshold_config.type in [
                ThresholdType.topk,
                ThresholdType.topk_hard,
        ]:
            for key, mask in mask_dict.items():
                if self.threshold_config.value >= mask.numel():
                    if self.threshold_config.type != ThresholdType.topk:
                        mask_dict[key] = torch.ones_like(mask)
                    continue

                value, index = torch.topk(
                    mask.flatten(),
                    k=self.threshold_config.value,
                )

                out = torch.zeros_like(mask.flatten())
                if self.threshold_config.type == ThresholdType.topk:
                    out[index] = value
                else:
                    out[index] = 1.0
                mask_dict[key] = out.reshape(mask.size())

        else:
            raise NotImplementedError

        # Update the explanation with the thresholded masks:
        for key, mask in mask_dict.items():
            explanation[key] = mask

        return explanation
