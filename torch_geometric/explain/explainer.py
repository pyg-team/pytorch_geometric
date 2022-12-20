import warnings
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import (
    ExplainerAlgorithm,
    Explanation,
    HeteroExplanation,
)
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ThresholdConfig,
)
from torch_geometric.typing import EdgeType, NodeType


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
    def get_prediction(self, *args, **kwargs) -> Tensor:
        r"""Returns the prediction of the model on the input graph.

        If the model mode is :obj:`"regression"`, the prediction is returned as
        a scalar value.
        If the model mode is :obj:`"multiclass_classification"` or
        :obj:`"binary_classification"`, the prediction is returned as the
        predicted class label.

        Args:
            *args: Arguments passed to the model.
            **kwargs (optional): Additional keyword arguments passed to the
                model.
        """
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            out = self.model(*args, **kwargs)

        self.model.train(training)

        return out

    def get_masked_prediction(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        node_mask: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        edge_mask: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
        **kwargs,
    ) -> Tensor:
        r"""Returns the prediction of the model on the input graph with node
        and edge masks applied."""
        if isinstance(x, Tensor) and node_mask is not None:
            x = node_mask * x
        elif isinstance(x, dict) and node_mask is not None:
            x = {key: value * node_mask[key] for key, value in x.items()}

        if isinstance(edge_mask, Tensor):
            set_masks(self.model, edge_mask, edge_index, apply_sigmoid=False)
        elif isinstance(edge_mask, dict):
            set_hetero_masks(self.model, edge_mask, edge_index,
                             apply_sigmoid=False)

        out = self.get_prediction(x, edge_index, **kwargs)
        clear_masks(self.model)
        return out

    def __call__(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Optional[Tensor] = None,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Computes the explanation of the GNN for the given inputs and
        target.

        .. note::

            If you get an error message like "Trying to backward through the
            graph a second time", make sure that the target you provided
            was computed with :meth:`torch.no_grad`.

        Args:
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The input
                node features of a homogeneous or heterogeneous graph.
            edge_index (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The
                input edge indices of a homogeneous or heterogeneous graph.
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
        prediction: Optional[Tensor] = None
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
            prediction = self.get_prediction(x, edge_index, **kwargs)
            target = self.get_target(prediction)

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

        # Add explainer objectives to the `Explanation` object:
        explanation._model_config = self.model_config
        explanation.prediction = prediction
        explanation.target = target
        explanation.index = index
        explanation.target_index = target_index

        # Add model inputs to the `Explanation` object:
        if isinstance(explanation, Explanation):
            explanation._model_args = list(kwargs.keys())
            explanation.x = x
            explanation.edge_index = edge_index

            for key, arg in kwargs.items():  # Add remaining `kwargs`:
                explanation[key] = arg

        elif isinstance(explanation, HeteroExplanation):
            assert isinstance(x, dict)
            # TODO Add `explanation._model_args`
            for node_type, value in x.items():
                explanation[node_type].x = value

            assert isinstance(edge_index, dict)
            for edge_type, value in edge_index.items():
                explanation[edge_type].edge_index = value

            for key, arg in kwargs.items():  # Add remaining `kwargs`:
                if isinstance(arg, dict):
                    # Keyword arguments are likely named `{attr_name}_dict`
                    # while we only want to assign the `{attr_name}` to the
                    # `HeteroExplanation` object:
                    key = key[:-5] if key.endswith('_dict') else key
                    for type_name, value in arg.items():
                        explanation[type_name][key] = value
                else:
                    explanation[key] = arg

        explanation.validate_masks()
        return explanation.threshold(self.threshold_config)

    def get_target(self, prediction: Tensor) -> Tensor:
        r"""Returns the target of the model from a given prediction.

        If the model mode is of type :obj:`"regression"`, the prediction is
        returned as it is.
        If the model mode is of type :obj:`"multiclass_classification"` or
        :obj:`"binary_classification"`, the prediction is returned as the
        predicted class label.
        """
        if self.model_config.mode == ModelMode.binary_classification:
            # TODO: Allow customization of the thresholds used below.
            if self.model_config.return_type == ModelReturnType.raw:
                return (prediction > 0).long().view(-1)
            if self.model_config.return_type == ModelReturnType.probs:
                return (prediction > 0.5).long().view(-1)
            assert False

        if self.model_config.mode == ModelMode.multiclass_classification:
            return prediction.argmax(dim=-1)

        return prediction
