import re
import copy
import warnings
import torch
import torch_geometric
from torch_sparse import coalesce, SparseTensor
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

from ..utils.num_nodes import maybe_num_nodes

__num_nodes_warn_msg__ = (
    'The number of nodes in your data object can only be inferred by its {} '
    'indices, and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated nodes. Please consider explicitly setting '
    'the number of nodes for this data object by assigning it to '
    '`data.num_nodes`.')


def size_repr(value):
    if torch.is_tensor(value):
        return list(value.size())
    if isinstance(value, SparseTensor):
        return f'({list(value.sizes())}, nnz={value.nnz()})'
    elif isinstance(value, list) or isinstance(value, tuple):
        return [len(value)]
    elif isinstance(value, dict):
        return '{...}'
    else:
        return value


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        adj (SparseTensor, optional): Sparse adjacency matrix.
            (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes according to
            :attr:`edge_index`. (default: :obj:`None`)
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    Note that the data object is not restricted to these attributes and can be
    extented by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, adj=None, edge_index=None, edge_attr=None,
                 num_nodes=None, x=None, y=None, pos=None, norm=None,
                 face=None, **kwargs):
        self.__adj__ = adj
        self.__edge_index__ = edge_index
        self.__edge_attr__ = edge_attr
        self.__num_nodes__ = num_nodes
        self.x = x
        self.y = y
        self.pos = pos
        self.norm = norm
        self.face = face
        for key, item in kwargs.items():
            self[key] = item

        if torch_geometric.is_debug_enabled():
            self.debug()

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        if torch_geometric.is_debug_enabled():
            data.debug()

        return data

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def __private_keys__(self):
        return ['__adj__', '__edge_index__', '__edge_attr__']

    @property
    def __hidden_keys__(self):
        return ['__num_nodes__']

    @property
    def keys(self):
        r"""Returns all names of data attributes."""
        keys = [
            key for key in self.__dict__.keys()
            if key not in self.__hidden_keys__ and self[key] is not None
        ]
        keys = [i[2:-2] if i in self.__private_keys__ else i for i in keys]
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        if len(keys) == 0:
            keys = sorted(self.keys)
        else:
            keys = [key for key in keys if self[key] is not None]

        for key in keys:
            yield key, self[key]

    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if bool(re.search('(index|face)', key)):
            return 1
        else:
            return 0

    def __inc__(self, key, value):
        r""""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if bool(re.search('(index|face)', key)):
            return self.num_nodes
        else:
            return 0

    @property
    def adj(self):
        if self.__adj__ is None and self.__edge_index__ is not None:
            col, row = self.__edge_index__
            return SparseTensor(
                row=row, col=col, value=self.__edge_attr__,
                sparse_sizes=torch.Size([self.num_nodes, self.num_nodes]))
        return self.__adj__

    @adj.setter
    def adj(self, adj):
        self.__adj__ = adj

    @property
    def edge_index(self):
        if self.__edge_index__ is None and self.__adj__ is not None:
            col, row, _ = self.__adj__.coo()
            return torch.stack([row, col], dim=0)
        return self.__edge_index__

    @edge_index.setter
    def edge_index(self, edge_index):
        self.__edge_index__ = edge_index

    @property
    def edge_attr(self):
        if self.__edge_attr__ is None and self.__adj__ is not None:
            value = self.__adj__.storage.value()
            if value is not None:
                return value
        return self.__edge_attr__

    @edge_attr.setter
    def edge_attr(self, edge_attr):
        self.__edge_attr__ = edge_attr

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if self.__num_nodes__ is not None:
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        if self.__adj__ is not None:
            return self.__adj__.size(1)
        if self.__edge_index__ is not None:
            warnings.warn(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.__edge_index__)
        if self.face is not None:
            warnings.warn(__num_nodes_warn_msg__.format('face'))
            return maybe_num_nodes(self.face)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        if self.__adj__ is not None:
            return self.adj.nnz()
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.__edge_attr__ is not None:
            edge_attr = self.__edge_attr__
            return 1 if edge_attr.dim() == 1 else edge_attr.size(1)
        if self.__adj__ is not None and self.__adj__.has_value():
            value = self.__adj__.storage.value()
            return 1 if value.dim() == 1 else value.size(1)
        return 0

    def __is_adj_coalesced__(self):
        if self.__adj__ is not None:
            return self.adj.is_coalesced()
        return True

    def __is_edge_index_coalesced__(self):
        if self.__edge_index__ is not None:
            edge_index, _ = coalesce(self.__edge_index__, None, self.num_nodes,
                                     self.num_nodes)
            return self.__edge_index__.numel() == edge_index.numel() and (bool(
                (self.__edge_index__ == edge_index).all()))
        return True

    def is_coalesced(self):
        r"""Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        return (self.__is_adj_coalesced__()
                and self.__is_edge_index_coalesced__())

    def coalesce(self):
        r""""Orders and removes duplicated entries from edge indices."""
        if self.__adj__ is not None:
            self.__adj__ = self.__adj__.coalesce()
        if self.__edge_index__ is not None:
            self.__edge_index__, self.__edge_attr__ = coalesce(
                self.__edge_index__, self.__edge_attr__, self.num_nodes,
                self.num_nodes)
        return self

    def contains_isolated_nodes(self):
        r"""Returns :obj:`True`, if the graph contains isolated nodes."""
        return contains_isolated_nodes(self.edge_index, self.num_nodes)

    def contains_self_loops(self):
        """Returns :obj:`True`, if the graph contains self-loops."""
        return contains_self_loops(self.edge_index)

    def is_undirected(self):
        r"""Returns :obj:`True`, if graph edges are undirected."""
        return is_undirected(self.edge_index, self.edge_attr, self.num_nodes)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item) or isinstance(item, SparseTensor):
                self[key] = func(item)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(
            lambda x: x.contiguous() if torch.is_tensor(x) else x, *keys)

    @property
    def device(self):
        for _, item in self:
            if hasattr(item, 'device'):
                return item.device
        return torch.device('cpu')

    def to(self, device, *keys):
        r"""Performs tensor device conversion to all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) or isinstance(v, SparseTensor)
            else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })

    def debug(self):
        if self.__edge_index__ is not None:
            if self.edge_index.dtype != torch.long:
                raise RuntimeError(
                    ('Expected edge indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.edge_index.dtype))

        if self.face is not None:
            if self.face.dtype != torch.long:
                raise RuntimeError(
                    ('Expected face indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.face.dtype))

        if self.__edge_index__ is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    ('Edge indices should have shape [2, num_edges] but found'
                     ' shape {}').format(self.edge_index.size()))

        if self.__edge_index__ is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    ('Face indices should have shape [3, num_faces] but found'
                     ' shape {}').format(self.face.size()))

        if self.face is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Face indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.__edge_index__ is not None and self.__edge_attr__ is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    ('Edge indices and edge attributes hold a differing '
                     'number of edges, found {} and {}').format(
                         self.edge_index.size(), self.edge_attr.size()))

        if self.x is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node features should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.x.size(0)))

        if self.pos is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node positions should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.pos.size(0)))

        if self.norm is not None:
            if self.norm.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node normals should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.norm.size(0)))

    def __repr__(self):
        keys = [key for key in self.keys if key[:2] != '__']
        info = ['{}={}'.format(key, size_repr(self[key])) for key in keys]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))
