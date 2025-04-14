from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from torch_geometric.utils.mixin import CastMixin


class ExplanationType(Enum):
    """Enum class for the explanation type."""
    model = 'model'
    phenomenon = 'phenomenon'
    generative = 'generative'


class MaskType(Enum):
    """Enum class for the mask type."""
    object = 'object'
    common_attributes = 'common_attributes'
    attributes = 'attributes'


class ModelMode(Enum):
    """Enum class for the model return type."""
    binary_classification = 'binary_classification'
    multiclass_classification = 'multiclass_classification'
    regression = 'regression'


class ModelTaskLevel(Enum):
    """Enum class for the model task level."""
    node = 'node'
    edge = 'edge'
    graph = 'graph'


class ModelReturnType(Enum):
    """Enum class for the model return type."""
    raw = 'raw'
    probs = 'probs'
    log_probs = 'log_probs'


class ThresholdType(Enum):
    """Enum class for the threshold type."""
    hard = 'hard'
    topk = 'topk'
    topk_hard = 'topk_hard'
    # connected = 'connected'  # TODO


@dataclass
class ExplainerConfig(CastMixin):
    r"""Configuration class to store and validate high level explanation
    parameters.

    Args:
        explanation_type (ExplanationType or str): The type of explanation to
            compute. The possible values are:

                - :obj:`"model"`: Explains the model prediction.

                - :obj:`"phenomenon"`: Explains the phenomenon that the model
                  is trying to predict.

            In practice, this means that the explanation algorithm will either
            compute their losses with respect to the model output
            (:obj:`"model"`) or the target output (:obj:`"phenomenon"`).

        node_mask_type (MaskType or str, optional): The type of mask to apply
            on nodes. The possible values are (default: :obj:`None`):

                - :obj:`None`: Will not apply any mask on nodes.

                - :obj:`"object"`: Will mask each node.

                - :obj:`"common_attributes"`: Will mask each feature.

                - :obj:`"attributes"`: Will mask each feature across all nodes.

        edge_mask_type (MaskType or str, optional): The type of mask to apply
            on edges. Has the sample possible values as :obj:`node_mask_type`.
            (default: :obj:`None`)
    """
    explanation_type: ExplanationType
    node_mask_type: Optional[MaskType]
    edge_mask_type: Optional[MaskType]

    def __init__(
        self,
        explanation_type: Union[ExplanationType, str],
        node_mask_type: Optional[Union[MaskType, str]] = None,
        edge_mask_type: Optional[Union[MaskType, str]] = None,
    ):
        if node_mask_type is not None:
            node_mask_type = MaskType(node_mask_type)
        if edge_mask_type is not None:
            edge_mask_type = MaskType(edge_mask_type)

        if edge_mask_type is not None and edge_mask_type != MaskType.object:
            raise ValueError(f"'edge_mask_type' needs be None or of type "
                             f"'object' (got '{edge_mask_type.value}')")

        if node_mask_type is None and edge_mask_type is None:
            if ExplanationType(
                    explanation_type) is not ExplanationType.generative:
                raise ValueError("Either 'node_mask_type' or 'edge_mask_type' "
                                 "must be provided")

        self.explanation_type = ExplanationType(explanation_type)
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type


@dataclass
class ModelConfig(CastMixin):
    r"""Configuration class to store model parameters.

    Args:
        mode (ModelMode or str): The mode of the model. The possible values
            are:

                - :obj:`"binary_classification"`: A binary classification
                  model.

                - :obj:`"multiclass_classification"`: A multiclass
                  classification model.

                - :obj:`"regression"`: A regression model.

        task_level (ModelTaskLevel or str): The task-level of the model.
            The possible values are:

                - :obj:`"node"`: A node-level prediction model.

                - :obj:`"edge"`: An edge-level prediction model.

                - :obj:`"graph"`: A graph-level prediction model.

        return_type (ModelReturnType or str, optional): The return type of the
            model. The possible values are (default: :obj:`None`):

                - :obj:`"raw"`: The model returns raw values.

                - :obj:`"probs"`: The model returns probabilities.

                - :obj:`"log_probs"`: The model returns log-probabilities.
    """
    mode: ModelMode
    task_level: ModelTaskLevel
    return_type: ModelReturnType

    def __init__(
        self,
        mode: Union[ModelMode, str],
        task_level: Union[ModelTaskLevel, str],
        return_type: Optional[Union[ModelReturnType, str]] = None,
    ):
        self.mode = ModelMode(mode)
        self.task_level = ModelTaskLevel(task_level)

        if return_type is None and self.mode == ModelMode.regression:
            return_type = ModelReturnType.raw

        self.return_type = ModelReturnType(return_type)

        if (self.mode == ModelMode.regression
                and self.return_type != ModelReturnType.raw):
            raise ValueError(f"A model for regression needs to return raw "
                             f"outputs (got {self.return_type.value})")

        if (self.mode == ModelMode.binary_classification and self.return_type
                not in [ModelReturnType.raw, ModelReturnType.probs]):
            raise ValueError(
                f"A model for binary classification needs to return raw "
                f"outputs or probabilities (got {self.return_type.value})")


@dataclass
class ThresholdConfig(CastMixin):
    r"""Configuration class to store and validate threshold parameters.

    Args:
        threshold_type (ThresholdType or str): The type of threshold to apply.
            The possible values are:

                - :obj:`None`: No threshold is applied.

                - :obj:`"hard"`: A hard threshold is applied to each mask.
                  The elements of the mask with a value below the :obj:`value`
                  are set to :obj:`0`, the others are set to :obj:`1`.

                - :obj:`"topk"`: A soft threshold is applied to each mask.
                  The top obj:`value` elements of each mask are kept, the
                  others are set to :obj:`0`.

                - :obj:`"topk_hard"`: Same as :obj:`"topk"` but values are set
                  to :obj:`1` for all elements which are kept.

        value (int or float, optional): The value to use when thresholding.
            (default: :obj:`None`)
    """
    type: ThresholdType
    value: Union[float, int]

    def __init__(
        self,
        threshold_type: Union[ThresholdType, str],
        value: Union[float, int],
    ):
        self.type = ThresholdType(threshold_type)
        self.value = value

        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Threshold value must be a float or int "
                             f"(got {type(self.value)}).")

        if (self.type == ThresholdType.hard
                and (self.value < 0 or self.value > 1)):
            raise ValueError(f"Threshold value must be between 0 and 1 "
                             f"(got {self.value})")

        if self.type in [ThresholdType.topk, ThresholdType.topk_hard]:
            if not isinstance(self.value, int):
                raise ValueError(f"Threshold value needs to be an integer "
                                 f"(got {type(self.value)}).")
            if self.value <= 0:
                raise ValueError(f"Threshold value needs to be positive "
                                 f"(got {self.value}).")
