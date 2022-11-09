from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from torch_geometric.utils.mixin import CastMixin


class ThresholdType(Enum):
    """Enum class for the threshold type."""
    none = 'none'
    hard = 'hard'
    topk = 'topk'
    topk_hard = 'topk_hard'
    # connected = 'connected'  # TODO


class ExplanationType(Enum):
    """Enum class for the explanation type."""
    model = 'model'
    phenomenon = 'phenomenon'


class MaskType(Enum):
    """Enum class for the mask type."""
    object = 'object'
    attributes = 'attributes'
    none = 'none'
    both = 'both'


class ModelMode(Enum):
    """Enum class for the model return type."""
    regression = 'regression'
    classification = 'classification'


class ModelReturnType(Enum):
    """Enum class for the model return type."""
    log_probs = 'log_probs'
    probs = 'probs'
    raw = 'raw'


class ModelTaskLevel(Enum):
    """Enum class for the model task level."""
    graph = 'graph'
    node = 'node'
    edge = 'edge'


@dataclass
class ThresholdConfig(CastMixin):
    r"""Class to store and validate threshold parameters.

    Args:
        type (ThresholdType or str): the type of threshold to apply.
            The possible values are:

                - none: no threshold is applied.

                - hard: a hard threshold is applied to each mask. The elements
                  of the mask with a value below the :obj:`value` are set to
                  :obj:`0`, the others are set to :obj:`1`.

                - topk: a soft threshold is applied to each mask. The top
                  obj:`value` elements of each mask are kept, the others are
                  set to :obj:`0`.

                - topk_hard: same as :obj:`"topk"` but values are set to
                  :obj:`1` for all elements which are kept.

        value (int or float, optional): The value to use when thresholding.
            (default: :obj:`None`)
    """
    type: ThresholdType
    value: Union[float, int]

    def __init__(
        self,
        threshold_type: Union[ThresholdType, str],
        value: Optional[Union[float, int]] = None,
    ):
        self.type = ThresholdType(threshold_type)
        self.value = value

        if self.type == ThresholdType.none:
            self.value = 0

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


@dataclass
class ExplainerConfig(CastMixin):
    r"""Class to store and validate high level explanation parameters.

    Attributes:
        explanation_type (ExplanationType or str): The type of explanation to
            compute. Can be either :obj:`"model"` or :obj:`"phenomenon"`.
            :obj:`"model"` will explain the model prediction, while
            :obj:`"phenomenon"` will explain the phenomenon that the model is
            trying to predict.

            .. note::

                In practice this means that the explanation algorithm will
                either compute their losses with respect to the model output
                or the target output.

        node_mask_type (MaskType, optional): The type of mask to apply on
            nodes. Can be :obj:`None`, :obj:`"object"`, :obj:`"attributes"` or
            :obj:`"both"`. :obj:`None` will not apply any mask on nodes,
            :obj:`"object"` will mask the whole node, :obj:`"attributes"` will
            ask the node attributes, and :obj:`"both"` will mask both the node
            and its attributes. (default: :obj:`None`)
        edge_mask_type (MaskType): The type of mask to apply on edges.
            Same types as :obj:`node_mask_type`.
    """
    explanation_type: ExplanationType
    node_mask_type: MaskType
    edge_mask_type: MaskType

    def __init__(
        self,
        explanation_type: Union[ExplanationType, str],
        node_mask_type: Optional[Union[MaskType, str]] = None,
        edge_mask_type: Optional[Union[MaskType, str]] = None,
    ):
        if node_mask_type is None:
            node_mask_type = MaskType.none
        if edge_mask_type is None:
            edge_mask_type = MaskType.none

        self.explanation_type = ExplanationType(explanation_type)
        self.node_mask_type = MaskType(node_mask_type)
        self.edge_mask_type = MaskType(edge_mask_type)

        if (self.node_mask_type == MaskType.none
                and self.edge_mask_type == MaskType.none):
            raise ValueError("Either 'node_mask_type' or 'edge_mask_type' "
                             "needs to be provided.")


@dataclass
class ModelConfig(CastMixin):
    # TODO Add doc-string.
    mode: ModelMode
    task_level: ModelTaskLevel
    return_type: ModelReturnType

    def __init__(
        self,
        mode: Union[ModelMode, str],
        task_level: Union[ModelTaskLevel, str],
        return_type: Union[ModelReturnType, str],
    ):
        self.mode = ModelMode(mode)
        self.task_level = ModelTaskLevel(task_level)
        self.return_type = ModelReturnType(return_type)

        if (self.mode == ModelMode.regression
                and self.return_type != ModelReturnType.raw):
            raise ValueError(f"A model for regression needs to return raw "
                             f"outputs (got {self.return_type.value})")
