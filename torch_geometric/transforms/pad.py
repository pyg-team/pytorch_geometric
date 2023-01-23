import numbers
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType, NodeType

PadValTypes = (int, float)


class Padding(ABC):
    r"""An abstract class for specifying padding values."""
    @abstractmethod
    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        pass


class MappingPadding(Padding):
    r"""An abstract class for specifying different padding values."""
    _key_types = ()
    _val_types = ()

    def __init__(
        self,
        padding_values: Optional[Dict[Any, Any]] = None,
        default_value: Union[int, float] = 0.0,
    ):
        padding_values = padding_values or {}
        if not isinstance(padding_values, dict):
            raise ValueError(f"Expected 'padding_values' to be a dictionary "
                             f"(got '{type(padding_values)}'")

        self.default_value = default_value
        self.padding_values = {
            key: UniformPadding(val) if isinstance(val, (int, float)) else val
            for key, val in padding_values.items()
        }

        for key, val in self.padding_values.items():
            if not isinstance(key, self._key_types):
                raise ValueError(f"Expected '{key}' to be of type "
                                 f"{self._key_types} (got '{type(key)}')")
            if not isinstance(val, self._val_types):
                raise ValueError(f"Expected the value of '{key}' to be of "
                                 f"type {self._val_types} (got '{type(val)}')")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'(padding_values={self.padding_values}, ' \
               f'default_value={self.default_value})'


@dataclass(init=False)
class UniformPadding(Padding):
    r"""Uniform padding independent of attribute name or node/edge type.

    Args:
        value (int or float, optional): The value to be used for padding.
            (default: :obj:`0.0`)
    """
    value: Union[int, float] = 0.0

    def __init__(self, value: Union[int, float] = 0.0):
        self.value = value

        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Expected 'value' to be an integer or float "
                             f"(got '{type(value)}'")

    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        return self.value


class AttrNamePadding(MappingPadding):
    r"""Indicates specific padding values for different attribute names. To use
    with :class:`Pad` transform.

    Example:

    .. code-block::

        padding = AttrNamePadding({'x': 3.0, 'y': 0.0}, default_value=1.0)
        transform = Pad(max_num_nodes, node_pad_value=padding)

    Args:
        padding_values (dict, optional): The mapping from attribute names to
            padding values. If an attribute is not specified in the mapping,
            the :obj:`default_value` value is used. (default: :obj:`None`)
        default_value (int or float, optional): The padding value used for
            attributes not specified in the :obj:`padding_values` mapping.
            (default: :obj:`0.0`)
    """
    _key_types = (str, )
    _val_types = (UniformPadding, *PadValTypes)

    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        if attr_name in self.padding_values.keys():
            return self.padding_values[attr_name].get_value()
        return self.default_value


class NodeTypePadding(MappingPadding):
    r"""Indicates specific padding values for different types of nodes. To use
    with :class:`Pad` transform for :class:`~torch_geometric.data.HeteroData`
    objects.

    Example:

    .. code-block::

        p1 = AttrNamePadding({'x': 3.0, 'y': 0.0})
        p2 = 3.0
        padding = NodeTypePadding({'v1': p1, 'v2': p2}, default_value=1.0)
        transform = Pad(max_num_nodes, node_pad_value=padding)

    Args:
        padding_values (dict, optional): The mapping from node types to
            padding values. If a node type is not specified in the mapping,
            the :obj:`default_value` value is used. (default: :obj:`None`)
        default_value (int or float, optional): The padding value used for
            node types not specified in the :obj:`padding_values` mapping.
            (default: :obj:`0.0`)
    """
    _key_types = (NodeType, )
    _val_types = (UniformPadding, AttrNamePadding, *PadValTypes)

    def get_value(
        self,
        store_type: Optional[NodeType] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        if store_type in self.padding_values.keys():
            return self.padding_values[store_type].get_value(None, attr_name)
        return self.default_value


class EdgeTypePadding(MappingPadding):
    r"""Indicates specific padding values for different types of edges. To use
    with :class:`Pad` transform for :class:`~torch_geometric.data.HeteroData`
    objects.

    Example:

    .. code-block::

        p1 = AttrNamePadding({'edge_x': 3.0, 'edge_y': 0.0})
        p2 = -4.0
        padding = EdgeTypePadding(
            {('v1', 'e1', 'v2'): p1, ('v2', 'e1', 'v1'): p2},
            default_value=1.0)
        transform = Pad(max_num_nodes, edge_pad_value=padding)

    Args:
        padding_values (dict, optional): The mapping from edge types to
            padding values. If an edge type is not specified in the mapping,
            the :obj:`default_value` value is used. (default: :obj:`None`)
        default_value (int or float, optional): The padding value used for
            edge types not specified in the :obj:`padding_values` mapping.
            (default: :obj:`0.0`)
    """
    _key_types = (tuple, )
    _val_types = (UniformPadding, AttrNamePadding, *PadValTypes)

    def __init__(
        self,
        padding_values: Optional[Dict[EdgeType,
                                      Union[int, float, UniformPadding,
                                            AttrNamePadding]]] = None,
        default_value=0.0,
    ):
        super().__init__(padding_values, default_value)

        for key in self.padding_values.keys():
            if not len(key) == 3:
                raise ValueError(f"Edge type '{key}'' must be a tuple of "
                                 f"length 3 (got {len(key)})")

    def get_value(
        self,
        store_type: Optional[EdgeType] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        if store_type in self.padding_values.keys():
            return self.padding_values[store_type].get_value(None, attr_name)
        return self.default_value


@functional_transform('pad')
class Pad(BaseTransform):
    r"""Applies padding to enforce consistent tensor shapes
    (functional name: :obj:`pad`).

    This transform will pad node and edge features up to a maximum allowed size
    in the node or edge feature dimension. By default :obj:`0.0` is used as the
    padding value and can be configured by setting :obj:`node_pad_value` and
    :obj:`edge_pad_value`.

    In case of applying :class:`Pad` to a :class:`~torch_geometric.data.Data`
    object, the :obj:`node_pad_value` value (or :obj:`edge_pad_value`) can be
    either:

    * an int, float or object of :class:`UniformPadding` class for cases when
      all attributes are going to be padded with the same value;
    * an object of :class:`AttrNamePadding` class for cases when padding is
      going to differ based on attribute names.

    In case of applying :class:`Pad` to a
    :class:`~torch_geometric.data.HeteroData` object, the :obj:`node_pad_value`
    value (or :obj:`edge_pad_value`) can be either:

    * an int, float or object of :class:`UniformPadding` class for cases when
      all attributes of all node (or edge) stores are going to be padded with
      the same value;
    * an object of :class:`AttrNamePadding` class for cases when padding is
      going to differ based on attribute names (but not based on node or edge
      types);
    * an object of class :class:`NodeTypePadding` or :class:`EdgeTypePadding`
      for cases when padding values are going to differ based on node or edge
      types. Padding values can also differ based on attribute names for a
      given node or edge type by using :class:`AttrNamePadding` objects as
      values of its `padding_values` argument.

    Note that in order to allow for consistent padding across all graphs in a
    dataset, below conditions must be met:

    * if :obj:`max_num_nodes` is a single value, it must be greater than or
      equal to the maximum number of nodes of any graph in the dataset;
    * if :obj:`max_num_nodes` is a dictionary, value for every node type must
      be greater than or equal to the maximum number of this type nodes of any
      graph in the dataset.

    Example below shows how to create a :class:`Pad` transform for an
    :class:`~torch_geometric.data.HeteroData` object. The object is padded to
    have :obj:`10` nodes of type :obj:`v0`, :obj:`20` nodes of type :obj:`v1`
    and :obj:`30` nodes of type :obj:`v2`.
    It is padded to have :obj:`80` edges of type :obj:`('v0', 'e0', 'v1')`.
    All the attributes of the :obj:`v0` nodes are padded using a value of
    :obj:`3.0`.
    The :obj:`x` attribute of the :obj:`v1` node type is padded using a value
    of :obj:`-1.0`, and the other attributes of this node type are padded using
    a value of :obj:`0.5`.
    All the attributes of node types other than :obj:`v0` and :obj:`v1` are
    padded using a value of :obj:`1.0`.
    All the attributes of the :obj:`('v0', 'e0', 'v1')` edge type are padded
    usin a value of :obj:`3.5`.
    The :obj:`edge_attr` attributes of the :obj:`('v1', 'e0', 'v0')` edge type
    are padded using a value of :obj:`-1.5`, and any other attributes of this
    edge type are padded using a value of :obj:`5.5`.
    All the attributes of edge types other than these two are padded using a
    value of :obj:`1.5`.

    Example:

    .. code-block::

        num_nodes = {'v0': 10, 'v1': 20, 'v2':30}
        num_edges = {('v0', 'e0', 'v1'): 80}

        node_padding = NodeTypePadding({
            'v0': 3.0,
            'v1': AttrNamePadding({'x': -1.0}, default_value=0.5),
        }, default_value=1.0)

        edge_padding = EdgeTypePadding({
            ('v0', 'e0', 'v1'): 3.5,
            ('v1', 'e0', 'v0'): AttrNamePadding({'edge_attr': -1.5},
                                                default_value=5.5),
        }, default_value=1.5)

        transform = Pad(num_nodes, num_edges, node_padding, edge_padding)

    Args:
        max_num_nodes (int or dict): The number of nodes after padding.
            In heterogeneous graphs, may also take in a dictionary denoting the
            number of nodes for specific node types.
        max_num_edges (int or dict, optional): The number of edges after
            padding.
            In heterogeneous graphs, may also take in a dictionary denoting the
            number of edges for specific edge types. (default: :obj:`None`)
        node_pad_value (int or float or Padding, optional): The fill value to
            use for node features. (default: :obj:`0.0`)
        edge_pad_value (int or float or Padding, optional): The fill value to
            use for edge features. (default: :obj:`0.0`)
            The :obj:`edge_index` tensor is padded with with the index of the
            first padded node (which represents a set of self-loops on the
            padded node). (default: :obj:`0.0`)
        mask_pad_value (bool, optional): The fill value to use for
            :obj:`train_mask`, :obj:`val_mask` and :obj:`test_mask` attributes
            (default: :obj:`False`).
        exclude_keys (list or tuple, optional): Keys to be removed
            from the input data object. (default: :obj:`None`)
    """
    def __init__(
        self,
        max_num_nodes: Union[int, Dict[NodeType, int]],
        max_num_edges: Optional[Union[int, Dict[EdgeType, int]]] = None,
        node_pad_value: Optional[Union[int, float, Padding]] = None,
        edge_pad_value: Optional[Union[int, float, Padding]] = None,
        mask_pad_value: bool = False,
        exclude_keys: Optional[Union[List[str], Tuple[str]]] = None,
    ):
        self._default_pad_value = UniformPadding(0.0)
        self.max_num_nodes = self._NumNodes(max_num_nodes)
        self.max_num_edges = self._NumEdges(max_num_edges, self.max_num_nodes)

        self.exclude_keys = set(
            exclude_keys) if exclude_keys is not None else set()
        assert 'x' not in self.exclude_keys, \
            'Cannot create a `Pad` with `x` attribute being excluded.'

        self.node_pad = self._process_padding_argument(node_pad_value,
                                                       'node_pad_value')
        self.edge_pad = self._process_padding_argument(edge_pad_value,
                                                       'edge_pad_value')

        if mask_pad_value is None:
            mask_pad_value = 0.0
        mask_attrs = ['train_mask', 'val_mask', 'test_mask']
        self.node_additional_attrs_pad = {
            key: mask_pad_value
            for key in mask_attrs
        }

    def _process_padding_argument(self, padding: Any, name: str) -> Padding:
        if isinstance(padding, Padding):
            return padding
        if padding is None:
            return self._default_pad_value
        try:
            return UniformPadding(padding)
        except AssertionError as e:
            raise AssertionError(f'Invalid type of `{name}`.') from e

    class _IntOrDict(ABC):
        def __init__(self, value):
            self.value = value
            self.is_number = isinstance(value, numbers.Number)

        @abstractmethod
        def get_value(self, key: Optional[Any] = None) -> None:
            pass

        def is_none(self):
            return self.value is None

    class _NumNodes(_IntOrDict):
        def __init__(self, value) -> None:
            assert isinstance(value, (int, dict)), \
                f'Parameter `max_num_nodes` must be of type int or dict ' \
                f'but is {type(value)}.'
            super().__init__(value)

        def get_value(self, key: Optional[NodeType] = None) -> int:
            if self.is_number or self.value is None:
                # Homodata case.
                return self.value

            assert isinstance(key, str)
            assert key in self.value.keys(), \
                f'The number of {key} nodes was not specified for padding.'
            return self.value[key]

    class _NumEdges(_IntOrDict):
        def __init__(self, value: Union[int, Dict[EdgeType, int], None],
                     num_nodes: '_NumNodes') -> None:  # noqa: F821
            assert value is None or isinstance(value, (int, dict)), \
                f'If provided, parameter `max_num_edges` must be of type ' \
                f'int or dict but is {type(value)}.'

            if value is not None:
                if isinstance(value, int):
                    num_edges = value
                else:
                    num_edges = defaultdict(lambda: defaultdict(int))
                    for k, v in value.items():
                        src_node, edge_type, dst_node = k
                        num_edges[src_node, dst_node][edge_type] = v
            elif num_nodes.is_number:
                num_edges = num_nodes.get_value() * num_nodes.get_value()
            else:
                num_edges = defaultdict(lambda: defaultdict(int))

            self.num_nodes = num_nodes
            super().__init__(num_edges)

        def get_value(self, key: Optional[EdgeType] = None) -> int:
            if self.is_number or self.value is None:
                # Homodata case.
                return self.value

            assert isinstance(key, tuple) and len(key) == 3
            src_v, edge_type, dst_v = key
            if (src_v, dst_v) in self.value.keys():
                if edge_type in self.value[src_v, dst_v]:
                    return self.value[src_v, dst_v][edge_type]

            max_num_edges = self.num_nodes.get_value(
                src_v) * self.num_nodes.get_value(dst_v)
            self.value[src_v, dst_v][edge_type] = max_num_edges
            return self.value[src_v, dst_v][edge_type]

    def __should_pad_node_attr(self, attr_name: str) -> bool:
        if attr_name in self.node_additional_attrs_pad:
            return True
        if self.exclude_keys is None or attr_name not in self.exclude_keys:
            return True
        return False

    def __should_pad_edge_attr(self, attr_name: str) -> bool:
        if self.max_num_edges.is_none():
            return False
        if attr_name == 'edge_index':
            return True
        if self.exclude_keys is None or attr_name not in self.exclude_keys:
            return True
        return False

    def __get_node_padding(
            self, attr_name: str,
            node_type: Optional[NodeType] = None) -> Union[int, float]:
        if attr_name in self.node_additional_attrs_pad:
            return self.node_additional_attrs_pad[attr_name]
        return self.node_pad.get_value(node_type, attr_name)

    def __get_edge_padding(
            self, attr_name: str,
            edge_type: Optional[EdgeType] = None) -> Union[int, float]:
        return self.edge_pad.get_value(edge_type, attr_name)

    def __call__(self, data: Union[Data,
                                   HeteroData]) -> Union[Data, HeteroData]:
        if isinstance(data, Data):
            assert isinstance(self.node_pad,
                              (UniformPadding, AttrNamePadding)), \
                f'Node padding for Data objects must be of type ' \
                f'UniformPadding or AttrNamePadding and is ' \
                f'{type(self.node_pad)}'
            assert isinstance(self.edge_pad,
                              (UniformPadding, AttrNamePadding)), \
                f'Edge padding for Data objects must be of type ' \
                f'UniformPadding or AttrNamePadding and is ' \
                f'{type(self.edge_pad)}'

            assert self.max_num_nodes.is_number, \
                f'Number of nodes for Data objects must be a number and is ' \
                f'of type {type(self.max_num_nodes.value)}'
            assert self.max_num_edges.is_number or \
                   self.max_num_edges.is_none(), \
                   'Number of edges for Data objects must be a number or None.'

            for store in data.stores:
                for key in self.exclude_keys:
                    if key in store.keys():
                        del store[key]
                self.__pad_edge_store(store, data.__cat_dim__, data.num_nodes)
                self.__pad_node_store(store, data.__cat_dim__)
        else:
            assert isinstance(
                self.node_pad,
                (UniformPadding, AttrNamePadding, NodeTypePadding))
            assert isinstance(
                self.edge_pad,
                (UniformPadding, AttrNamePadding, EdgeTypePadding))
            for edge_type, store in data.edge_items():
                for key in self.exclude_keys:
                    if key in store.keys():
                        del store[key]
                src_node_type, _, dst_node_type = edge_type
                self.__pad_edge_store(store, data.__cat_dim__,
                                      (data[src_node_type].num_nodes,
                                       data[dst_node_type].num_nodes),
                                      edge_type)
            for node_type, store in data.node_items():
                for key in self.exclude_keys:
                    if key in store.keys():
                        del store[key]
                self.__pad_node_store(store, data.__cat_dim__, node_type)
        return data

    def __pad_node_store(self, store: NodeStorage, get_dim_fn: Callable,
                         node_type: Optional[str] = None) -> None:
        attrs_to_pad = [
            attr for attr in store.keys()
            if store.is_node_attr(attr) and self.__should_pad_node_attr(attr)
        ]
        if not attrs_to_pad:
            return
        num_target_nodes = self.max_num_nodes.get_value(node_type)
        assert num_target_nodes > store.num_nodes, \
            f'The number of nodes after padding ({num_target_nodes}) must ' \
            f'be greater than the number of nodes in the data object ' \
            f'({store.num_nodes}).'
        num_pad_nodes = num_target_nodes - store.num_nodes

        for attr_name in attrs_to_pad:
            attr = store[attr_name]
            pad_value = self.__get_node_padding(attr_name, node_type)
            dim = get_dim_fn(attr_name, attr)
            store[attr_name] = self._pad_tensor_dim(attr, dim, num_pad_nodes,
                                                    pad_value)

    def __pad_edge_store(self, store: EdgeStorage, get_dim_fn: Callable,
                         num_nodes: Union[int, Tuple[int, int]],
                         edge_type: Optional[str] = None) -> None:
        attrs_to_pad = set(
            attr for attr in store.keys()
            if store.is_edge_attr(attr) and self.__should_pad_edge_attr(attr))
        if not attrs_to_pad:
            return
        num_target_edges = self.max_num_edges.get_value(edge_type)
        assert num_target_edges >= store.num_edges, \
            f'The number of edges after padding ({num_target_edges}) cannot ' \
            f'be lower than the number of edges in the data object ' \
            f'({store.num_edges}).'
        num_pad_edges = num_target_edges - store.num_edges

        if isinstance(num_nodes, tuple):
            src_pad_value, dst_pad_value = num_nodes
        else:
            src_pad_value = dst_pad_value = num_nodes

        for attr_name in attrs_to_pad:
            attr = store[attr_name]
            dim = get_dim_fn(attr_name, attr)
            if attr_name == 'edge_index':
                store[attr_name] = self._pad_edge_index(
                    attr, num_pad_edges, src_pad_value, dst_pad_value)
            else:
                pad_value = self.__get_edge_padding(attr_name, edge_type)
                store[attr_name] = self._pad_tensor_dim(
                    attr, dim, num_pad_edges, pad_value)

    @staticmethod
    def _pad_tensor_dim(input: torch.Tensor, dim: int, length: int,
                        pad_value: float) -> torch.Tensor:
        r"""Pads the input tensor in the specified dim with a constant value of
        the given length.
        """
        pads = [0] * (2 * input.ndim)
        pads[-2 * dim - 1] = length
        return F.pad(input, pads, 'constant', pad_value)

    @staticmethod
    def _pad_edge_index(input: torch.Tensor, length: int, src_pad_value: float,
                        dst_pad_value: float) -> torch.Tensor:
        r"""Pads the edges :obj:`edge_index` feature with values specified
        separately for src and dst nodes.
        """
        pads = [0, length, 0, 0]
        padded = F.pad(input, pads, 'constant', src_pad_value)
        if src_pad_value != dst_pad_value:
            padded[1, input.shape[1]:] = dst_pad_value
        return padded

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        s += f'max_num_nodes={self.max_num_nodes.value}, '
        s += f'max_num_edges={self.max_num_edges.value}, '
        s += f'node_pad_value={self.node_pad}, '
        s += f'edge_pad_value={self.edge_pad})'
        return s
