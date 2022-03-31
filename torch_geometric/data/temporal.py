import copy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data.data import BaseData, size_repr
from torch_geometric.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)


class TemporalData(BaseData):
    r"""A data object composed by a stream of events describing a temporal
    graph.
    The :class:`~torch_geometric.data.TemporalData` object can hold a list of
    events (that can be understood as temporal edges in a graph) with
    structured messages.
    An event is composed by a source node, a destination node, a timestamp
    and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
    represented with these four values.

    In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
    the behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.

    .. code-block:: python

        from torch import Tensor
        from torch_geometric.data import TemporalData

        events = TemporalData(
            src=Tensor([1,2,3,4]),
            dst=Tensor([2,3,4,5]),
            t=Tensor([1000,1010,1100,2000]),
            msg=Tensor([1,1,0,0])
        )

        # Add additional arguments to `events`:
        events.y = Tensor([1,1,0,0])

        # It is also possible to set additional arguments in the constructor
        events = TemporalData(
            ...,
            y=Tensor([1,1,0,0])
        )

        # Get the number of events:
        events.num_events
        >>> 4

        # Analyzing the graph structure:
        events.num_nodes
        >>> 5

        # PyTorch tensor functionality:
        events = events.pin_memory()
        events = events.to('cuda:0', non_blocking=True)

    Args:
        src (Tensor, optional): A list of source nodes for the events with
            shape :obj:`[num_events]`. (default: :obj:`None`)
        dst (Tensor, optional): A list of destination nodes for the events
            with shape :obj:`[num_events]`. (default: :obj:`None`)
        t (Tensor, optional): The timestamps for each event with shape
            :obj:`[num_events]`. (default: :obj:`None`)
        msg (Tensor, optional): Messages feature matrix with shape
            :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.

    .. note::
        The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
        of :obj`msg` should be the same (:obj:`num_events`).
    """
    def __init__(
        self,
        src: Optional[Tensor] = None,
        dst: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        msg: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg

        for key, value in kwargs.items():
            setattr(self, key, value)

    def index_select(self, idx: Any) -> 'TemporalData':
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if value.size(0) == self.num_events:
                data[key] = value[idx]
        return data

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, str):
            return self._store[idx]
        return self.index_select(idx)

    def __setitem__(self, key: str, value: Any):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __iter__(self) -> Iterable:
        for i in range(self.num_events):
            yield self[i]

    def __len__(self) -> int:
        return self.num_events

    def __call__(self, *args: List[str]) -> Iterable:
        for key, value in self._store.items(*args):
            yield key, value

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def stores_as(self, data: 'TemporalData'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def debug(self):
        pass  # TODO

    @property
    def num_nodes(self) -> int:
        r"""Returns the number of nodes in the graph."""
        return max(int(self.src.max()), int(self.dst.max())) + 1

    @property
    def num_events(self) -> int:
        r"""Returns the number of events loaded.

        .. note::
            In a :class:`~torch_geometric.data.TemporalData`, each row denotes
            an event.
            Thus, they can be also understood as edges.
        """
        return self.src.size(0)

    @property
    def num_edges(self) -> int:
        r"""Alias for :meth:`~torch_geometric.data.TemporalData.num_events`."""
        return self.num_events

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        size = (int(self.src.max()), int(self.dst.max()))
        return size if dim is None else size[dim]

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif key in ['src', 'dst']:
            return self.num_nodes
        else:
            return 0

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ', '.join([size_repr(k, v) for k, v in self._store.items()])
        return f'{cls}({info})'

    ###########################################################################

    def train_val_test_split(self, val_ratio: float = 0.15,
                             test_ratio: float = 0.15):
        r"""Splits the data in training, validation and test sets according to
        time.

        Args:
            val_ratio (float, optional): The proportion (in percents) of the
                dataset to include in the validation split.
                (default: :obj:`0.15`)
            test_ratio (float, optional): The proportion (in percents) of the
                dataset to include in the test split. (default: :obj:`0.15`)
        """
        val_time, test_time = np.quantile(
            self.t.cpu().numpy(),
            [1. - val_ratio - test_ratio, 1. - test_ratio])

        val_idx = int((self.t <= val_time).sum())
        test_idx = int((self.t <= test_time).sum())

        return self[:val_idx], self[val_idx:test_idx], self[test_idx:]

    ###########################################################################

    def coalesce(self):
        raise NotImplementedError

    def has_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        raise NotImplementedError


###############################################################################


def prepare_idx(idx):
    if isinstance(idx, int):
        return slice(idx, idx + 1)
    if isinstance(idx, (list, tuple)):
        return torch.tensor(idx)
    elif isinstance(idx, slice):
        return idx
    elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
        return idx
    elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
        return idx

    raise IndexError(
        f"Only strings, integers, slices (`:`), list, tuples, and long or "
        f"bool tensors are valid indices (got '{type(idx).__name__}')")
