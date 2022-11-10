from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from torch_geometric.utils.mixin import CastMixin


class ExplanationType(Enum):
    """Enum class for the explanation type."""
    model = 'model'
    phenomenon = 'phenomenon'


class MaskType(Enum):
    """Enum class for the mask type."""
    object = 'object'
    attributes = 'attributes'
    both = 'both'


class ModelMode(Enum):
    """Enum class for the model return type."""
    classification = 'classification'
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

            In practice this means that the explanation algorithm will either
            compute their losses with respect to the model output or the target
            output.

        node_mask_type (MaskType or str, optional): The type of mask to apply
            on nodes. The possible values are (default: :obj:`None`):

                - :obj:`None`: Will not apply any mask on nodes.

                - :obj:`"object"`: Will mask the whole node.

                - :obj:`"attributes"`: Will mask the node attributes.

                - :obj:`"both"`: Will mask both the node and its attributes..

        edge_mask_type (MaskType or str, optional): The type of mask to apply
            on edges. Same types as :obj:`node_mask_type`.
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

        self.explanation_type = ExplanationType(explanation_type)
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type

        if self.node_mask_type is None and self.edge_mask_type is None:
            raise ValueError("Either 'node_mask_type' or 'edge_mask_type' "
                             "must be provided.")


@dataclass
class ModelConfig(CastMixin):
    r"""Configuration class to store model parameters.

    Args:
        mode (ModelMode or str): The mode of the model. The possible values
            are:

                - :obj:`"classification"`: A classification model.

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

                - :obj:`"topk_hard"`: Aame as :obj:`"topk"` but values are set
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
