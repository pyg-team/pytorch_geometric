import copy
import functools
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import ExplainerAlgorithm, Explanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
    ThresholdConfig,
    ThresholdType,
)


def decorate_link_prediction_model(model, index, model_config: ModelConfig):
    if model_config.return_type == ModelReturnType.raw:

        def get_complement_and_out(out):
            out = out.sigmoid()
            complement = 1 - out
            return torch.stack((complement, out), dim=-1)

        model_config.return_type = ModelReturnType.probs
    elif model_config.return_type == ModelReturnType.log_probs:

        def get_complement_and_out(out):
            complement = torch.log(1 + torch.exp((torch.log(-out))))
            return torch.stack((complement, out), dim=-1)
    elif model_config.return_type == ModelReturnType.probs:

        def get_complement_and_out(out):
            complement = 1 - out
            return torch.stack((complement, out), dim=-1)
    else:
        raise NotImplementedError

    @functools.wraps(model)
    def wrapper(*args, **kwargs):
        out = model(*args, edge_label_index=index, **kwargs)
        out = get_complement_and_out(out)
        return out

    return wrapper


class Explainer:
    r"""An explainer class for instance-level explanations of Graph Neural
    Networks.

    Args:
        model (torch.nn.Module): The model to explain.
        algorithm (ExplainerAlgorithm): The explanation algorithm.
        explainer_config (ExplainerConfig): The explainer configuration.
        model_config (ModelConfig): The model configuration.
        threshold_config (ThresholdConfig, optional): The threshold
            configuration. (default: :obj:`None`)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        algorithm: ExplainerAlgorithm,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.model = model
        self.algorithm = algorithm

        self.explainer_config = ExplainerConfig.cast(explainer_config)
        self.model_config = ModelConfig.cast(model_config)
        self.threshold_config = ThresholdConfig.cast(threshold_config)

        if not self.algorithm.supports(
                self.explainer_config,
                self.model_config,
        ):
            raise ValueError(
                f"The explanation algorithm "
                f"'{self.algorithm.__class__.__name__}' does not support the "
                f"given explanation settings.")

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
        with torch.no_grad():
            out = self.model(*args, **kwargs)
        if self.model_config.mode == ModelMode.classification:
            return out.argmax(dim=-1)
        return out

    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Optional[Tensor] = None,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[Union[int, Tensor]] = None,
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
            target_index (int or torch.Tensor, optional): The target indices to
                explain in case targets are multi-dimensional.
                (default: :obj:`None`)
            **kwargs: additional arguments to pass to the GNN.
        """
        if self.model_config.task_level == ModelTaskLevel.edge:
            # decorate the model forward function to make its output match the
            # node classification format
            self.model.forward = decorate_link_prediction_model(
                self.model.forward, index, self.model_config)
            assert isinstance(index, Tensor) and len(index.shape) == 1 or (
                len(index.shape) == 2 and index.shape[0] == 1
            ), "Explainer only supports single edge index expressed as a "
            "tensor for now"
            index = 0

        # Choose the `target` depending on the explanation type:
        if (self.explainer_config.explanation_type ==
                ExplanationType.phenomenon):
            if target is None:
                raise ValueError(
                    f"The target has to be provided for the explanation type "
                    f"'{self.explainer_config.explanation_type.value}'")
        else:
            target = self.get_prediction(x=x, edge_index=edge_index, **kwargs)

        explanation = self.algorithm(
            model=self.model,
            x=x,
            edge_index=edge_index,
            explainer_config=self.explainer_config,
            model_config=self.model_config,
            target=target,
            index=index,
            target_index=target_index,
            **kwargs,
        )

        if self.model_config.task_level == ModelTaskLevel.edge:
            # restore original forward function in the model
            self.model.forward = self.model.forward.__wrapped__

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
