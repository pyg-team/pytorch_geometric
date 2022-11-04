from dataclasses import dataclass
from enum import Enum
from typing import Union

__all__ = [
    "ThresholdType",
    "ExplanationType",
    "MaskType",
    "Threshold",
    "ExplainerConfig",
    "ModelConfig",
]


class ThresholdType(Enum):
    """Enum class for treshold type."""
    none = "none"
    hard = "hard"
    topk = "topk"
    connected = "connected"
    topk_hard = "topk_hard"


class ExplanationType(Enum):
    """Enum class for explanation type."""
    model = "model"
    phenomenon = "phenomenon"


class MaskType(Enum):
    """Enum class for mask type."""
    object = "object"
    attributes = "attributes"
    none = "none"
    both = "both"


class ModelMode(Enum):
    """Enum class for model return type."""
    regression = "regression"
    classification = "classification"


class ModelReturnType(Enum):
    """Enum class for model return type."""
    logits = "logits"
    probs = "probs"
    raw = "raw"
    regression = "regression"


class ModelTaskLevel(Enum):
    """Enum class for model task level."""
    graph = "graph"
    node = "node"
    edge = "edge"


@dataclass
class Threshold:
    """Class to store and validate threshold parameters.

    Attributes:
        type (ThresholdType): the type of threshold to apply.
            The possible values are:

                - none: no threshold is applied.

                - hard: a hard threshold is applied to each mask. The elements
                  of the mask with a value below the :obj:`value` are set to
                  :obj:`0`, the others are set to :obj:`1`.

                - topk: a soft threshold is applied to each mask. the top
                  obj;`value` elements of each mask are kept and the
                  rest is set  to :obj:`0`.

                - topk_hard: same as topk but the elements are set to :obj:`1`

                - connected: Not implemented yet. should returned the connected
                  subgraph size :obj:`value` that contribute the most to the
                  prediction.

        value (Union[float,int]): the value to use when thresholding.
    """
    type: ThresholdType
    value: Union[float, int]

    def __init__(self, type: Union[ThresholdType, str], value: Union[float,
                                                                     int]):
        self.type = ThresholdType(type)
        self.value = value
        if self.type == ThresholdType.none:
            self.value = 0
        self.__post_init__()

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError("Threshold value must be a float or int.")

        if self.type == ThresholdType.hard:
            if self.value < 0 or self.value > 1:
                raise ValueError(f'Invalid threshold value {self.value}. '
                                 f'Valid values are in [0, 1].')

        if self.type in [
                ThresholdType.topk, ThresholdType.topk_hard,
                ThresholdType.connected
        ]:
            if self.value <= 0 or not isinstance(self.value, int):
                raise ValueError(f'Invalid threshold value {self.value}. '
                                 f'Valid values are positif integers.')


@dataclass
class ExplainerConfig:
    """Class to store and validate high level explanation parameters.

    Attributes:
        explanation_type (ExplanationType): type of explanation to compute.
            Can be either "model" or "phenomenon". Model will explain the model
            prediction, while phenomenon will explain the phenomenon that the
            model is trying to predict.

            .. note::

                In practice this means that the explanation algorithm will
                either compute their losses with respect to the model output
                or the target output.

        node_mask_type (MaskType): type of mask to apply on nodes. Can be
            :obj:`none`, :obj:`object`, :obj:`attributes` or :obj:`both`. None
            will not apply any mask on nodes, object will mask the whole node,
            attributes will mask the node attributes and both will mask both
            the node and its attributes.
        edge_mask_type (MaskType): type of mask to apply on edges. Same types
            as :obj:`node_mask_type`.
    """
    explanation_type: ExplanationType
    node_mask_type: MaskType
    edge_mask_type: MaskType

    def __init__(self, explanation_type: Union[ExplanationType, str],
                 node_mask_type: Union[MaskType, str],
                 edge_mask_type: Union[MaskType, str]) -> None:
        self.explanation_type = ExplanationType(explanation_type)
        self.node_mask_type = MaskType(node_mask_type)
        self.edge_mask_type = MaskType(edge_mask_type)


@dataclass
class ModelConfig:
    return_type: Union[str, ModelReturnType]
    task_level: Union[str, ModelTaskLevel]
    mode: Union[str, ModelMode]

    def __init__(
        self,
        return_type: Union[str, ModelReturnType],
        task_level: Union[str, ModelTaskLevel] = ModelTaskLevel.graph,
        mode: Union[str, ModelMode] = ModelMode.classification,
    ) -> None:
        self.return_type = ModelReturnType(return_type)
        self.task_level = ModelTaskLevel(task_level)
        self.mode = ModelMode(mode)
        self.__post_init__()

    def __post_init__(self):
        if self.task_level == ModelReturnType.regression:
            self.return_type = ModelReturnType.regression
