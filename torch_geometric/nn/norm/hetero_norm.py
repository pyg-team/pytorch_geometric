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
    mean_funcs = {"batchnorm":batch_mean, "instancenorm":instance_mean, "layernorm":layer_mean}
    var_funcs = {"batchnorm":batch_var, "instancenorm":instance_var, "layernorm":layer_var}
    accepted_norm_types = ["batchnorm", "instancenorm", "layernorm"]

    def __init__(self, in_channels: Dict[str, int], norm_type: str,
                 types: Optional[Union[List[NodeType],List[EdgeType]]] = None,
                 eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True,
                 allow_single_element: bool = False):
        super().__init__()
        if not norm_type.lower() in accepted_norm_types:
            raise ValueError('Please choose norm type from "BatchNorm", "InstanceNorm", "LayerNorm"')
        self.norm_type = norm_type
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
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.allow_single_element = allow_single_element
        self.affine = affine
        self.in_channels = in_channels
        self.hetero_linear = HeteroDictLinear(self.in_channels, self.in_channels, self.types, **kwargs)
        self.means = ParameterDict({mean_type:torch.zeros(self.in_channels) for mean_type in self.types})
        self.vars = ParameterDict({var_type:torch.ones(self.in_channels) for var_type in self.types})
        self.allow_single_element = allow_single_element
        self.mean_func = mean_funcs[self.norm_type]
        self.var_func = var_funcs[self.norm_type]
        self.reset_parameters()

    @classmethod
    def from_homogeneous(self, norm_module):
        self.norm_type = None
        for norm_type in accepted_norm_types:
            if norm_type in str(norm_module).lower(): 
                self.norm_type = norm_type 
        if self.norm_type is None:
            raise ValueError('Please only pass one of "BatchNorm", "InstanceNorm", "LayerNorm"')
        try:
            # pyg norms
            self.in_channels = norm_module.in_channels
        except AttributeError:
            try:
                # torch native batch/instance norm
                self.in_channels = norm_module.num_features
            except AttributeError:
                # torch native layer norm
                self.in_channels = norm_module.normalized_shape
                if not isinstance(self.in_channels, int):
                    raise ValueError("If passing torch.nn.LayerNorm, \
                        please ensure that `normalized_shape` is a single integer")
        self.eps = norm_module.eps
        try:
            # store batch/instance norm
            self.momentum = norm_module.momentum
            self.track_running_stats = norm_module.track_running_stats
        except AttributeError:
            # layer norm
            self.momentum = None
            self.track_running_stats = False

        try:
            self.affine = norm_module.affine
        except AttributeError:
            self.affine = norm_module.elementwise_affine

        self.allow_single_element = hasattr(norm_module, "allow_single_element")
        self.mean_func = mean_funcs[self.norm_type]
        self.var_func = var_funcs[self.norm_type]


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hetero_linear.reset_parameters()
        for type_i in self.types:
            self.means[type_i] = torch.zeros(self.in_channels)
            self.vars[type_i] = torch.ones(self.in_channels)

    def fused_forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        out = x.new_empty(x.size(0), self.in_channels)
        x_dict = {self.types[i]:x[type_vec == i] for i in range(len(self.types))}
        return dict_forward(x_dict)

    def dict_forward(
        self,
        x_dict: Dict[Union[NodeType, EdgeType], Tensor],
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:
        out_dict = {}
        for x_type, x in x_dict.items():
            out_dict[x_type] = x - self.mean_func(x) / torch.sqrt(self.var_func(x) + self.eps)
        return self.hetero_linear(out_dict)

    def forward(
        self,
        x: Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]],
        type_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]]:

        if isinstance(x, dict):
            return self.dict_forward(x)

        elif isinstance(x, Tensor) and type_vec is not None:
            return self.fused_forward(x, type_vec)

        raise ValueError(f"Encountered invalid forward types in "
                         f"'{self.__class__.__name__}'")

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'
