import json
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType, EdgeType, NodeType
from torch_geometric.utils import sort_edge_index


class LocalGraphStore(GraphStore):
    r"""This class implements the :class:`torch_geometric.data.GraphStore`
    interface to act as a local graph store for distributed training.
    """
    def __init__(self):
        super().__init__()
        self._edge_index: Dict[Tuple, EdgeTensorType] = {}
        self._edge_attr: Dict[Tuple, EdgeAttr] = {}
        self._edge_id: Dict[Tuple, Tensor] = {}

        self.num_partitions = 1
        self.partition_idx = 0
        # Mapping between node ID and partition ID
        self.node_pb: Union[Tensor, Dict[NodeType, Tensor]] = None
        # Mapping between edge ID and partition ID
        self.edge_pb: Union[Tensor, Dict[EdgeType, Tensor]] = None
        # Meta information related to partition and graph store info
        self.meta: Optional[Dict[Any, Any]] = None
        # If data is sorted based on destination nodes (CSC format):
        self.is_sorted: Optional[bool] = None

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value)

    def get_partition_ids_from_nids(
        self,
        ids: torch.Tensor,
        node_type: Optional[NodeType] = None,
    ) -> Tensor:
        r"""Get the partition IDs of node IDs for a specific node type."""
        return self.node_pb[ids]

    def get_partition_ids_from_eids(self, eids: torch.Tensor,
                                    edge_type: Optional[EdgeType] = None):
        r"""Get the partition IDs of edge IDs for a specific edge type."""
        return self.edge_pb[eids]

    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        self._edge_id[self.key(edge_attr)] = edge_id
        return True

    def get_edge_id(self, *args, **kwargs) -> Optional[EdgeTensorType]:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id.get(self.key(edge_attr))

    def remove_edge_id(self, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id.pop(self.key(edge_attr), None) is not None

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self._edge_index[self.key(edge_attr)] = edge_index
        self._edge_attr[self.key(edge_attr)] = edge_attr
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self._edge_index.get(self.key(edge_attr), None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        self._edge_attr.pop(self.key(edge_attr), None)
        return self._edge_index.pop(self.key(edge_attr), None) is not None

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return [self._edge_attr[key] for key in self._edge_index.keys()]

    # Initialization ##########################################################

    @classmethod
    def from_data(
        cls,
        edge_id: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        is_sorted: bool = False,
    ) -> 'LocalGraphStore':
        r"""Creates a local graph store from a homogeneous or heterogenous
        :pyg:`PyG` graph.

        Args:
            edge_id (torch.Tensor): The global identifier for every local edge.
            edge_index (torch.Tensor): The local edge indices.
            num_nodes (int): The number of nodes in the local graph.
            is_sorted (bool): Indicate if edge_index is sorted on col/dst_node
                (CSC format). (default: :obj:`False`)
        """
        graph_store = cls()
        graph_store.meta = {'is_hetero': False}

        if not is_sorted:
            edge_index, edge_id = sort_edge_index(
                edge_index,
                edge_id,
                sort_by_row=False,
            )

        attr = dict(
            edge_type=None,
            layout='coo',
            size=(num_nodes, num_nodes),
            is_sorted=True,
        )

        graph_store.put_edge_index(edge_index, **attr)
        graph_store.put_edge_id(edge_id, **attr)

        return graph_store

    @classmethod
    def from_hetero_data(
        cls,
        edge_id_dict: Dict[EdgeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
        is_sorted: bool = False,
    ) -> "LocalGraphStore":
        r"""Creates a local graph store from a heterogeneous :pyg:`PyG` graph.

        Args:
            edge_id_dict (Dict[EdgeType, torch.Tensor]): The global identifier
                for every local edge of every edge type.
            edge_index_dict (Dict[EdgeType, torch.Tensor]): The local edge
                indices of every edge type.
            num_nodes_dict: (Dict[str, int]): The number of nodes for every
                node type.
            is_sorted (bool): Indicate if edge_index is sorted on col/dst_node
                (CSC format)
        """
        graph_store = cls()
        graph_store.meta = {'is_hetero': True}

        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            attr = dict(
                edge_type=edge_type,
                layout='coo',
                size=(num_nodes_dict[src], num_nodes_dict[dst]),
                is_sorted=True,
            )
            edge_id = edge_id_dict[edge_type]
            if not is_sorted:
                edge_index, edge_id = sort_edge_index(
                    edge_index,
                    edge_id,
                    sort_by_row=False,
                )
            graph_store.put_edge_index(edge_index, **attr)
            graph_store.put_edge_id(edge_id, **attr)
        return graph_store

    @classmethod
    def from_partition(cls, root: str, pid: int) -> 'LocalGraphStore':
        with open(osp.join(root, 'META.json'), 'r') as f:
            meta = json.load(f)

        part_dir = osp.join(root, f'part_{pid}')
        assert osp.exists(part_dir)

        graph_data = torch.load(osp.join(part_dir, 'graph.pt'))
        graph_store = cls()
        graph_store.is_sorted = meta['is_sorted']

        if not meta['is_hetero']:
            edge_index = torch.stack((graph_data['row'], graph_data['col']),
                                     dim=0)
            edge_id = graph_data['edge_id']
            if not graph_store.is_sorted:
                edge_index, edge_id = sort_edge_index(edge_index, edge_id,
                                                      sort_by_row=False)

            attr = dict(edge_type=None, layout='coo', size=graph_data['size'],
                        is_sorted=True)
            graph_store.put_edge_index(edge_index, **attr)
            graph_store.put_edge_id(edge_id, **attr)

        if meta['is_hetero']:
            for edge_type, data in graph_data.items():
                attr = dict(
                    edge_type=edge_type,
                    layout='coo',
                    size=data['size'],
                    is_sorted=True,
                )
                edge_index = torch.stack((data['row'], data['col']), dim=0)
                edge_id = data['edge_id']

                if not graph_store.is_sorted:
                    edge_index, edge_id = sort_edge_index(
                        edge_index, edge_id, sort_by_row=False)
                graph_store.put_edge_index(edge_index, **attr)
                graph_store.put_edge_id(edge_id, **attr)

        return graph_store
