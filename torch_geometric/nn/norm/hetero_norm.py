import torch
from torch import Tensor
from torch_geometric.nn.to_hetero_module import ToHeteroLinear
from torch_geometric.nn.typing import Metadata
from torch_geometric.nn.parameter_dict import ParameterDict
from typing import Dict

class HeteroNorm(torch.nn.Module):
    r"""Applies normalization over node features for each node type using:
    BatchNorm <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.BatchNorm.html#torch_geometric.nn.norm.BatchNorm>,
    InstanceNorm <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.InstanceNorm.html#torch_geometric.nn.norm.InstanceNorm>,
    or LayerNorm <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.LayerNorm.html#torch_geometric.nn.norm.LayerNorm>

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample.
            Use :obj:`-1` for lazy initialization.
            If passing as an int, types is required as well.
        norm_type (str): Which of "BatchNorm", "InstanceNorm", "LayerNorm" to use
            (default: "BatchNorm")
        types (List[str], optional): Only needed if in_channels
            is passed as an int.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        allow_single_element (bool, optional): If set to :obj:`True`, batches
            with only a single element will work as during in evaluation.
            That is the running mean and variance will be used.
            Requires :obj:`track_running_stats=True`. (default: :obj:`False`)
    """
    def __init__(self, in_channels: Dict[str, int], norm_type: str,
                 types: Optional[Union[List[NodeType],List[EdgeType]]] = None,
                 eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True,
                 allow_single_element: bool = False):
        super().__init__()
        if not norm_type.lower() in ["batchnorm", "instancenorm", "layernorm"]:
            raise ValueError('Please choose norm type from "BatchNorm", "InstanceNorm", "LayerNorm"')
        if allow_single_element and not track_running_stats:
            raise ValueError("'allow_single_element' requires "
                             "'track_running_stats' to be set to `True`")
        if isinstance(in_channels, dict):
            self.types = list(in_channels.keys())
            if any([int(i) == -1 for i in in_channels.values()]):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)
            if types is not None and self.types != types:
                raise ValueError("User provided `types` list does not match \
                 the keys of the `in_channels` dictionary")
        else:
            if in_channels == -1:
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)
            self.types = types
            if self.types is None:
                raise ValueError("Please provide a list of types if \
                    passing `in_channels` as an int")
            in_channels = {node_type: in_channels for node_type in self.types}
        self.in_channels = in_channels
        self.hetero_linear = HeteroDictLinear(self.in_channels, self.in_channels, self.types, **kwargs)
        self.means = ParameterDict({mean_type:torch.zeros(self.in_channels) for mean_type in self.types})
        self.vars = ParameterDict({var_type:torch.ones(self.in_channels) for var_type in self.types})
        self.allow_single_element = allow_single_element
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hetero_linear.reset_parameters()
        for type_i in self.types:
            self.means[type_i] = torch.zeros(self.in_channels)
            self.vars[type_i] = torch.ones(self.in_channels)

    def forward(
        self,
        x_dict: Dict[Union[NodeType, EdgeType], Tensor],
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input
                features for each individual type.
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'
