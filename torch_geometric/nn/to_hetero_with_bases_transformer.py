from typing import List, Dict, Union, Optional, Any
from torch_geometric.typing import NodeType, EdgeType, Metadata

import copy
import warnings

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_sparse import SparseTensor

from torch_geometric.nn.dense import Linear
from torch_geometric.nn.fx import Transformer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.hetero import get_unused_node_types

try:
    from torch.fx import GraphModule, Graph, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


def to_hetero_with_bases(module: Module, metadata: Metadata, num_bases: int,
                         in_channels: Optional[Dict[str, int]] = None,
                         input_map: Optional[Dict[str, str]] = None,
                         debug: bool = False) -> GraphModule:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent
    via the basis-decomposition technique introduced in the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper:
    For this, the heterogeneous graph is mapped to a typed homogeneous graph,
    in which its feature representations are aligned and grouped to a single
    representation.
    All GNN layers inside the model will then perform message passing via
    basis-decomposition regularization.
    This transformation is especially useful in highly multi-relational data,
    such that the number of parameters no longer depend on the number of
    relations of the input graph:

    .. code-block:: python

        import torch
        from torch_geometric.nn import SAGEConv, to_hetero_with_bases

        class GNN(torch.nn.Module):
            def __init__(self):
                self.conv1 = SAGEConv((16, 16), 32)
                self.conv2 = SAGEConv((32, 32), 32)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                return x

        model = GNN()

        node_types = ['paper', 'author']
        edge_types = [
            ('paper', 'cites', 'paper'),
            ('paper' 'written_by', 'author'),
            ('author', 'writes', 'paper'),
        ]
        metadata = (node_types, edge_types)

        model = to_hetero_with_bases(model, metadata, num_bases=3,
                                     in_channels={'x': 16})
        model(x_dict, edge_index_dict)

    where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
    hold node features and edge connectivity information for each node type and
    edge type, respectively.
    In case :obj:`in_channels` is given for a specific input argument, its
    heterogeneous feature information is first aligned to the given
    dimensionality.

    The below illustration shows the original computation graph of the
    homogeneous model on the left, and the newly obtained computation graph of
    the regularized heterogeneous model on the right:

    .. figure:: ../_figures/to_hetero_with_bases.svg
      :align: center
      :width: 90%

      Transforming a model via :func:`to_hetero_with_bases`.

    Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
    :math:`f_{\theta}^{(\ell)}` is duplicated :obj:`num_bases` times and
    stored in a set :math:`\{ f_{\theta}^{(\ell, b)} : b \in \{ 1, \ldots, B \}
    \}` (one instance for each basis in
    :obj:`num_bases`), and message passing in layer :math:`\ell` is performed
    via

    .. math::

        \mathbf{h}^{(\ell)}_v = \sum_{r \in \mathcal{R}} \sum_{b=1}^B
        f_{\theta}^{(\ell, b)} ( \mathbf{h}^{(\ell - 1)}_v, \{
        a^{(\ell)}_{r, b} \cdot \mathbf{h}^{(\ell - 1)}_w :
        w \in \mathcal{N}^{(r)}(v) \}),

    where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
    \mathcal{V}` under relation :math:`r \in \mathcal{R}`.
    Notably, only the trainable basis coefficients :math:`a^{(\ell)}_{r, b}`
    depend on the relations in :math:`\mathcal{R}`.

    Args:
        module (torch.nn.Module): The homogeneous model to transform.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        num_bases (int): The number of bases to use.
        in_channels (Dict[str, int], optional): A dictionary holding
            information about the desired input feature dimensionality of
            input arguments of :obj:`module.forward`.
            In case :obj:`in_channels` is given for a specific input argument,
            its heterogeneous feature information is first aligned to the given
            dimensionality.
            This allows handling of node and edge features with varying feature
            dimensionality across different types. (default: :obj:`None`)
        input_map (Dict[str, str], optional): A dictionary holding information
            about the type of input arguments of :obj:`module.forward`.
            For example, in case :obj:`arg` is a node-level argument, then
            :obj:`input_map['arg'] = 'node'`, and
            :obj:`input_map['arg'] = 'edge'` otherwise.
            In case :obj:`input_map` is not further specified, will try to
            automatically determine the correct type of input arguments.
            (default: :obj:`None`)
        debug: (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToHeteroWithBasesTransformer(module, metadata, num_bases,
                                               in_channels, input_map, debug)
    return transformer.transform()


class ToHeteroWithBasesTransformer(Transformer):
    def __init__(
        self,
        module: Module,
        metadata: Metadata,
        num_bases: int,
        in_channels: Optional[Dict[str, int]] = None,
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        super().__init__(module, input_map, debug)

        unused_node_types = get_unused_node_types(*metadata)
        if len(unused_node_types) > 0:
            warnings.warn(
                f"There exist node types ({unused_node_types}) whose "
                f"representations do not get updated during message passing "
                f"as they do not occur as destination type in any edge type. "
                f"This may lead to unexpected behaviour.")

        self.metadata = metadata
        self.num_bases = num_bases
        self.in_channels = in_channels or {}
        assert len(metadata) == 2
        assert len(metadata[0]) > 0 and len(metadata[1]) > 0

        # Compute IDs for each node and edge type:
        self.node_type2id = {k: i for i, k in enumerate(metadata[0])}
        self.edge_type2id = {k: i for i, k in enumerate(metadata[1])}

    def transform(self) -> GraphModule:
        self._node_offset_dict_initialized = False
        self._edge_offset_dict_initialized = False
        self._edge_type_initialized = False
        out = super().transform()
        del self._node_offset_dict_initialized
        del self._edge_offset_dict_initialized
        del self._edge_type_initialized
        return out

    def placeholder(self, node: Node, target: Any, name: str):
        if node.type is not None:
            Type = EdgeType if self.is_edge_level(node) else NodeType
            node.type = Dict[Type, node.type]

        out = node

        # Create `node_offset_dict` and `edge_offset_dict` dictionaries in case
        # they are not yet initialized. These dictionaries hold the cumulated
        # sizes used to create a unified graph representation and to split the
        # output data.
        if self.is_edge_level(node) and not self._edge_offset_dict_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function',
                                         target=get_edge_offset_dict,
                                         args=(node, self.edge_type2id),
                                         name='edge_offset_dict')
            self._edge_offset_dict_initialized = True

        elif not self._node_offset_dict_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function',
                                         target=get_node_offset_dict,
                                         args=(node, self.node_type2id),
                                         name='node_offset_dict')
            self._node_offset_dict_initialized = True

        # Create a `edge_type` tensor used as input to `HeteroBasisConv`:
        if self.is_edge_level(node) and not self._edge_type_initialized:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_function', target=get_edge_type,
                                         args=(node, self.edge_type2id),
                                         name='edge_type')
            self._edge_type_initialized = True

        # Add `Linear` operation to align features to the same dimensionality:
        if name in self.in_channels:
            self.graph.inserting_after(out)
            out = self.graph.create_node('call_module',
                                         target=f'align_lin__{name}',
                                         args=(node, ),
                                         name=f'{name}__aligned')
            self._state[out.name] = self._state[name]

            lin = LinearAlign(self.metadata[int(self.is_edge_level(node))],
                              self.in_channels[name])
            setattr(self.module, f'align_lin__{name}', lin)

        # Perform grouping of type-wise values into a single tensor:
        if self.is_edge_level(node):
            self.graph.inserting_after(out)
            out = self.graph.create_node(
                'call_function', target=group_edge_placeholder,
                args=(out if name in self.in_channels else node,
                      self.edge_type2id,
                      self.find_by_name('node_offset_dict')),
                name=f'{name}__grouped')
            self._state[out.name] = 'edge'

        else:
            self.graph.inserting_after(out)
            out = self.graph.create_node(
                'call_function', target=group_node_placeholder,
                args=(out if name in self.in_channels else node,
                      self.node_type2id), name=f'{name}__grouped')
            self._state[out.name] = 'node'

        self.replace_all_uses_with(node, out)

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        # Call the `HeteroBasisConv` wrapper instead instead of a single
        # message passing layer. We need to inject the `edge_type` as first
        # argument in order to do so.
        node.args = (self.find_by_name('edge_type'), ) + node.args

    def output(self, node: Node, target: Any, name: str):
        # Split the output to dictionaries, holding either node type-wise or
        # edge type-wise data.
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node) and self.is_edge_level(value):
                self.graph.inserting_before(node)
                return self.graph.create_node(
                    'call_function', target=split_output,
                    args=(value, self.find_by_name('edge_offset_dict')),
                    name=f'{value.name}__split')

                pass
            elif isinstance(value, Node):
                self.graph.inserting_before(node)
                return self.graph.create_node(
                    'call_function', target=split_output,
                    args=(value, self.find_by_name('node_offset_dict')),
                    name=f'{value.name}__split')

            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        if node.type is not None and isinstance(node.args[0], Node):
            output = node.args[0]
            Type = EdgeType if self.is_edge_level(output) else NodeType
            node.type = Dict[Type, node.type]
        else:
            node.type = None

        node.args = (_recurse(node.args[0]), )

    def init_submodule(self, module: Module, target: str) -> Module:
        if not isinstance(module, MessagePassing):
            return module

        # Replace each `MessagePassing` module by a `HeteroBasisConv` wrapper:
        return HeteroBasisConv(module, len(self.metadata[1]), self.num_bases)


###############################################################################


class HeteroBasisConv(torch.nn.Module):
    # A wrapper layer that applies the basis-decomposition technique to a
    # heterogeneous graph.
    def __init__(self, module: MessagePassing, num_relations: int,
                 num_bases: int):
        super().__init__()

        self.num_relations = num_relations
        self.num_bases = num_bases

        # We make use of a post-message computation hook to inject the
        # basis re-weighting for each individual edge type.
        # This currently requires us to set `conv.fuse = False`, which leads
        # to a materialization of messages.
        def hook(module, inputs, output):
            assert isinstance(module._edge_type, Tensor)
            if module._edge_type.size(0) != output.size(0):
                raise ValueError(
                    f"Number of messages ({output.size(0)}) does not match "
                    f"with the number of original edges "
                    f"({module._edge_type.size(0)}). Does your message "
                    f"passing layer create additional self-loops? Try to "
                    f"remove them via 'add_self_loops=False'")
            weight = module.edge_type_weight.view(-1)[module._edge_type]
            weight = weight.view([-1] + [1] * (output.dim() - 1))
            return weight * output

        params = list(module.parameters())
        device = params[0].device if len(params) > 0 else 'cpu'

        self.convs = torch.nn.ModuleList()
        for _ in range(num_bases):
            conv = copy.deepcopy(module)
            conv.fuse = False  # Disable `message_and_aggregate` functionality.
            # We learn a single scalar weight for each individual edge type,
            # which is used to weight the output message based on edge type:
            conv.edge_type_weight = Parameter(
                torch.Tensor(1, num_relations, device=device))
            conv.register_message_forward_hook(hook)
            self.convs.append(conv)

        if self.num_bases > 1:
            self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
            elif sum([p.numel() for p in conv.parameters()]) > 0:
                warnings.warn(
                    f"'{conv}' will be duplicated, but its parameters cannot "
                    f"be reset. To suppress this warning, add a "
                    f"'reset_parameters()' method to '{conv}'")
            torch.nn.init.xavier_uniform_(conv.edge_type_weight)

    def forward(self, edge_type: Tensor, *args, **kwargs) -> Tensor:
        out = None
        # Call message passing modules and perform aggregation:
        for conv in self.convs:
            conv._edge_type = edge_type
            res = conv(*args, **kwargs)
            del conv._edge_type
            out = res if out is None else out.add_(res)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_relations='
                f'{self.num_relations}, num_bases={self.num_bases})')


class LinearAlign(torch.nn.Module):
    # Aligns representions to the same dimensionality. Note that this will
    # create lazy modules, and as such requires a forward pass in order to
    # initialize parameters.
    def __init__(self, keys: List[Union[NodeType, EdgeType]],
                 out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleDict()
        for key in keys:
            self.lins[key2str(key)] = Linear(-1, out_channels, bias=False)

    def forward(
        self, x_dict: Dict[Union[NodeType, EdgeType], Tensor]
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:

        for key, x in x_dict.items():
            x_dict[key] = self.lins[key2str(key)](x)
        return x_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_relations={len(self.lins)}, '
                f'out_channels={self.out_channels})')


###############################################################################

# These methods are used in order to receive the cumulated sizes of input
# dictionaries. We make use of them for creating a unified homogeneous graph
# representation, as well as to split the final output data once again.


def get_node_offset_dict(
    input_dict: Dict[NodeType, Union[Tensor, SparseTensor]],
    type2id: Dict[NodeType, int],
) -> Dict[NodeType, int]:

    cumsum = 0
    out: Dict[NodeType, int] = {}
    for key in type2id.keys():
        out[key] = cumsum
        cumsum += input_dict[key].size(0)
    return out


def get_edge_offset_dict(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
) -> Dict[EdgeType, int]:

    cumsum = 0
    out: Dict[EdgeType, int] = {}
    for key in type2id.keys():
        out[key] = cumsum
        value = input_dict[key]
        if isinstance(value, SparseTensor):
            cumsum += value.nnz()
        elif value.dtype == torch.long and value.size(0) == 2:
            cumsum += value.size(-1)
        else:
            cumsum += value.size(0)
    return out


###############################################################################

# This method computes the edge type of the final homogeneous graph
# representation. It will be used in the `HeteroBasisConv` wrapper.


def get_edge_type(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
) -> Tensor:

    inputs = [input_dict[key] for key in type2id.keys()]
    outs = []

    for i, value in enumerate(inputs):
        if value.size(0) == 2 and value.dtype == torch.long:  # edge_index
            out = value.new_full((value.size(-1), ), i, dtype=torch.long)
        elif isinstance(value, SparseTensor):
            out = torch.full((value.nnz(), ), i, dtype=torch.long,
                             device=value.device())
        else:
            out = value.new_full((value.size(0), ), i, dtype=torch.long)
        outs.append(out)

    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


###############################################################################

# These methods are used to group the individual type-wise components into a
# unfied single representation.


def group_node_placeholder(input_dict: Dict[NodeType, Tensor],
                           type2id: Dict[NodeType, int]) -> Tensor:

    inputs = [input_dict[key] for key in type2id.keys()]
    return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)


def group_edge_placeholder(
    input_dict: Dict[EdgeType, Union[Tensor, SparseTensor]],
    type2id: Dict[EdgeType, int],
    offset_dict: Dict[NodeType, int] = None,
) -> Union[Tensor, SparseTensor]:

    inputs = [input_dict[key] for key in type2id.keys()]

    if len(inputs) == 1:
        return inputs[0]

    # In case of grouping a graph connectivity tensor `edge_index` or `adj_t`,
    # we need to increment its indices:
    elif inputs[0].size(0) == 2 and inputs[0].dtype == torch.long:
        if offset_dict is None:
            raise AttributeError(
                "Can not infer node-level offsets. Please ensure that there "
                "exists a node-level argument before the 'edge_index' "
                "argument in your forward header.")

        outputs = []
        for value, (src_type, _, dst_type) in zip(inputs, type2id):
            value = value.clone()
            value[0, :] += offset_dict[src_type]
            value[1, :] += offset_dict[dst_type]
            outputs.append(value)

        return torch.cat(outputs, dim=-1)

    elif isinstance(inputs[0], SparseTensor):
        if offset_dict is None:
            raise AttributeError(
                "Can not infer node-level offsets. Please ensure that there "
                "exists a node-level argument before the 'SparseTensor' "
                "argument in your forward header.")

        # For grouping a list of SparseTensors, we convert them into a
        # unified `edge_index` representation in order to avoid conflicts
        # induced by re-shuffling the data.
        rows, cols = [], []
        for value, (src_type, _, dst_type) in zip(inputs, type2id):
            col, row, value = value.coo()
            assert value is None
            rows.append(row + offset_dict[src_type])
            cols.append(col + offset_dict[dst_type])

        row = torch.cat(rows, dim=0)
        col = torch.cat(cols, dim=0)
        return torch.stack([row, col], dim=0)

    else:
        return torch.cat(inputs, dim=0)


###############################################################################

# This method is used to split the output tensors into individual type-wise
# components:


def split_output(
    output: Tensor,
    offset_dict: Union[Dict[NodeType, int], Dict[EdgeType, int]],
) -> Union[Dict[NodeType, Tensor], Dict[EdgeType, Tensor]]:

    cumsums = list(offset_dict.values()) + [output.size(0)]
    sizes = [cumsums[i + 1] - cumsums[i] for i in range(len(offset_dict))]
    outputs = output.split(sizes)
    return {key: output for key, output in zip(offset_dict, outputs)}


###############################################################################


def key2str(key: Union[NodeType, EdgeType]) -> str:
    key = '__'.join(key) if isinstance(key, tuple) else key
    return key.replace(' ', '_')
