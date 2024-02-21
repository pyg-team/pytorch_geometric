from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType, NodeType


class Padding(ABC):
    r"""An abstract class for specifying padding values."""
    @abstractmethod
    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        pass


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


@dataclass(init=False)
class MappingPadding(Padding):
    r"""An abstract class for specifying different padding values."""
    values: Dict[Any, Padding]
    default: UniformPadding

    def __init__(
        self,
        values: Dict[Any, Union[int, float, Padding]],
        default: Union[int, float] = 0.0,
    ):
        if not isinstance(values, dict):
            raise ValueError(f"Expected 'values' to be a dictionary "
                             f"(got '{type(values)}'")

        self.values = {
            key: UniformPadding(val) if isinstance(val, (int, float)) else val
            for key, val in values.items()
        }
        self.default = UniformPadding(default)

        for key, value in self.values.items():
            self.validate_key_value(key, value)

    def validate_key_value(self, key: Any, value: Any) -> None:
        pass


class AttrNamePadding(MappingPadding):
    r"""Padding dependent on attribute names.

    Args:
        values (dict): The mapping from attribute names to padding values.
        default (int or float, optional): The padding value to use for
            attribute names not specified in :obj:`values`.
            (default: :obj:`0.0`)
    """
    def validate_key_value(self, key: Any, value: Any) -> None:
        if not isinstance(key, str):
            raise ValueError(f"Expected the attribute name '{key}' to be a "
                             f"string (got '{type(key)}')")

        if not isinstance(value, UniformPadding):
            raise ValueError(f"Expected the value of '{key}' to be of "
                             f"type 'UniformPadding' (got '{type(value)}')")

    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        padding = self.values.get(attr_name, self.default)
        return padding.get_value()


class NodeTypePadding(MappingPadding):
    r"""Padding dependent on node types.

    Args:
        values (dict): The mapping from node types to padding values.
        default (int or float, optional): The padding value to use for node
            types not specified in :obj:`values`. (default: :obj:`0.0`)
    """
    def validate_key_value(self, key: Any, value: Any) -> None:
        if not isinstance(key, str):
            raise ValueError(f"Expected the node type '{key}' to be a string "
                             f"(got '{type(key)}')")

        if not isinstance(value, (UniformPadding, AttrNamePadding)):
            raise ValueError(f"Expected the value of '{key}' to be of "
                             f"type 'UniformPadding' or 'AttrNamePadding' "
                             f"(got '{type(value)}')")

    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        padding = self.values.get(store_type, self.default)
        return padding.get_value(attr_name=attr_name)


class EdgeTypePadding(MappingPadding):
    r"""Padding dependent on node types.

    Args:
        values (dict): The mapping from edge types to padding values.
        default (int or float, optional): The padding value to use for edge
            types not specified in :obj:`values`. (default: :obj:`0.0`)
    """
    def validate_key_value(self, key: Any, value: Any) -> None:
        if not isinstance(key, tuple):
            raise ValueError(f"Expected the edge type '{key}' to be a tuple "
                             f"(got '{type(key)}')")

        if len(key) != 3:
            raise ValueError(f"Expected the edge type '{key}' to hold exactly "
                             f"three elements (got {len(key)})")

        if not isinstance(value, (UniformPadding, AttrNamePadding)):
            raise ValueError(f"Expected the value of '{key}' to be of "
                             f"type 'UniformPadding' or 'AttrNamePadding' "
                             f"(got '{type(value)}')")

    def get_value(
        self,
        store_type: Optional[Union[NodeType, EdgeType]] = None,
        attr_name: Optional[str] = None,
    ) -> Union[int, float]:
        padding = self.values.get(store_type, self.default)
        return padding.get_value(attr_name=attr_name)


class _NumNodes:
    def __init__(
        self,
        value: Union[int, Dict[NodeType, int], None],
    ) -> None:
        self.value = value

    def get_value(self, key: Optional[NodeType] = None) -> Optional[int]:
        if self.value is None or isinstance(self.value, int):
            return self.value
        assert isinstance(key, str)
        return self.value[key]


class _NumEdges:
    def __init__(
        self,
        value: Union[int, Dict[EdgeType, int], None],
        num_nodes: _NumNodes,
    ) -> None:

        if value is None:
            if isinstance(num_nodes.value, int):
                value = num_nodes.value * num_nodes.value
            else:
                value = {}

        self.value = value
        self.num_nodes = num_nodes

    def get_value(self, key: Optional[EdgeType] = None) -> Optional[int]:
        if self.value is None or isinstance(self.value, int):
            return self.value

        assert isinstance(key, tuple) and len(key) == 3
        if key not in self.value:
            num_src_nodes = self.num_nodes.get_value(key[0])
            num_dst_nodes = self.num_nodes.get_value(key[-1])
            assert num_src_nodes is not None and num_dst_nodes is not None
            self.value[key] = num_src_nodes * num_dst_nodes

        return self.value[key]


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
      values of its `values` argument.

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
            'v1': AttrNamePadding({'x': -1.0}, default=0.5),
        }, default=1.0)

        edge_padding = EdgeTypePadding({
            ('v0', 'e0', 'v1'): 3.5,
            ('v1', 'e0', 'v0'): AttrNamePadding({'edge_attr': -1.5},
                                                default=5.5),
        }, default=1.5)

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
        add_pad_mask (bool, optional): If set to :obj:`True`, will attach
            node-level :obj:`pad_node_mask` and edge-level :obj:`pad_edge_mask`
            attributes to the output which indicates which elements in the data
            are real (represented by :obj:`True`) and which were added as a
            result of padding (represented by :obj:`False`).
            (default: :obj:`False`)
        exclude_keys ([str], optional): Keys to be removed
            from the input data object. (default: :obj:`None`)
    """
    def __init__(
        self,
        max_num_nodes: Union[int, Dict[NodeType, int]],
        max_num_edges: Optional[Union[int, Dict[EdgeType, int]]] = None,
        node_pad_value: Union[int, float, Padding] = 0.0,
        edge_pad_value: Union[int, float, Padding] = 0.0,
        mask_pad_value: bool = False,
        add_pad_mask: bool = False,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.max_num_nodes = _NumNodes(max_num_nodes)
        self.max_num_edges = _NumEdges(max_num_edges, self.max_num_nodes)

        self.node_pad: Padding
        if not isinstance(node_pad_value, Padding):
            self.node_pad = UniformPadding(node_pad_value)
        else:
            self.node_pad = node_pad_value

        self.edge_pad: Padding
        if not isinstance(edge_pad_value, Padding):
            self.edge_pad = UniformPadding(edge_pad_value)
        else:
            self.edge_pad = edge_pad_value

        self.node_additional_attrs_pad = {
            key: mask_pad_value
            for key in ['train_mask', 'val_mask', 'test_mask']
        }

        self.add_pad_mask = add_pad_mask
        self.exclude_keys = set(exclude_keys or [])

    def __should_pad_node_attr(self, attr_name: str) -> bool:
        if attr_name in self.node_additional_attrs_pad:
            return True
        if self.exclude_keys is None or attr_name not in self.exclude_keys:
            return True
        return False

    def __should_pad_edge_attr(self, attr_name: str) -> bool:
        if self.max_num_edges.value is None:
            return False
        if attr_name == 'edge_index':
            return True
        if self.exclude_keys is None or attr_name not in self.exclude_keys:
            return True
        return False

    def __get_node_padding(
        self,
        attr_name: str,
        node_type: Optional[NodeType] = None,
    ) -> Union[int, float]:
        if attr_name in self.node_additional_attrs_pad:
            return self.node_additional_attrs_pad[attr_name]
        return self.node_pad.get_value(node_type, attr_name)

    def __get_edge_padding(
        self,
        attr_name: str,
        edge_type: Optional[EdgeType] = None,
    ) -> Union[int, float]:
        return self.edge_pad.get_value(edge_type, attr_name)

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        if isinstance(data, Data):
            assert isinstance(self.node_pad, (UniformPadding, AttrNamePadding))
            assert isinstance(self.edge_pad, (UniformPadding, AttrNamePadding))

            for key in self.exclude_keys:
                del data[key]

            num_nodes = data.num_nodes
            assert num_nodes is not None
            self.__pad_edge_store(data._store, data.__cat_dim__, num_nodes)
            self.__pad_node_store(data._store, data.__cat_dim__)
            data.num_nodes = self.max_num_nodes.get_value()
        else:
            assert isinstance(
                self.node_pad,
                (UniformPadding, AttrNamePadding, NodeTypePadding))
            assert isinstance(
                self.edge_pad,
                (UniformPadding, AttrNamePadding, EdgeTypePadding))

            for edge_type, edge_store in data.edge_items():
                for key in self.exclude_keys:
                    del edge_store[key]

                src_node_type, _, dst_node_type = edge_type
                num_src_nodes = data[src_node_type].num_nodes
                num_dst_nodes = data[dst_node_type].num_nodes
                assert num_src_nodes is not None and num_dst_nodes is not None
                self.__pad_edge_store(edge_store, data.__cat_dim__,
                                      (num_src_nodes, num_dst_nodes),
                                      edge_type)

            for node_type, node_store in data.node_items():
                for key in self.exclude_keys:
                    del node_store[key]
                self.__pad_node_store(node_store, data.__cat_dim__, node_type)
                data[node_type].num_nodes = self.max_num_nodes.get_value(
                    node_type)

        return data

    def __pad_node_store(
        self,
        store: NodeStorage,
        get_dim_fn: Callable,
        node_type: Optional[NodeType] = None,
    ) -> None:

        attrs_to_pad = [key for key in store.keys() if store.is_node_attr(key)]

        if len(attrs_to_pad) == 0:
            return

        num_target_nodes = self.max_num_nodes.get_value(node_type)
        assert num_target_nodes is not None
        assert store.num_nodes is not None
        assert num_target_nodes >= store.num_nodes, \
            f'The number of nodes after padding ({num_target_nodes}) cannot ' \
            f'be lower than the number of nodes in the data object ' \
            f'({store.num_nodes}).'
        num_pad_nodes = num_target_nodes - store.num_nodes

        if self.add_pad_mask:
            pad_node_mask = torch.ones(num_target_nodes, dtype=torch.bool)
            pad_node_mask[store.num_nodes:] = False
            store.pad_node_mask = pad_node_mask

        for attr_name in attrs_to_pad:
            attr = store[attr_name]
            pad_value = self.__get_node_padding(attr_name, node_type)
            dim = get_dim_fn(attr_name, attr)
            store[attr_name] = self._pad_tensor_dim(attr, dim, num_pad_nodes,
                                                    pad_value)

    def __pad_edge_store(
        self,
        store: EdgeStorage,
        get_dim_fn: Callable,
        num_nodes: Union[int, Tuple[int, int]],
        edge_type: Optional[EdgeType] = None,
    ) -> None:

        attrs_to_pad = set(
            attr for attr in store.keys()
            if store.is_edge_attr(attr) and self.__should_pad_edge_attr(attr))
        if not attrs_to_pad:
            return
        num_target_edges = self.max_num_edges.get_value(edge_type)
        assert num_target_edges is not None
        assert num_target_edges >= store.num_edges, \
            f'The number of edges after padding ({num_target_edges}) cannot ' \
            f'be lower than the number of edges in the data object ' \
            f'({store.num_edges}).'
        num_pad_edges = num_target_edges - store.num_edges

        if self.add_pad_mask:
            pad_edge_mask = torch.ones(num_target_edges, dtype=torch.bool)
            pad_edge_mask[store.num_edges:] = False
            store.pad_edge_mask = pad_edge_mask

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
