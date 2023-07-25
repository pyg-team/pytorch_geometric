import json
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType, EdgeType, NodeType





class LocalGraphStore(GraphStore):
    r"""This class implements the :class:`torch_geometric.data.GraphStore`
    interface to act as a local graph store for distributed training."""
    def __init__(self):
        super().__init__()
        self._edge_index: Dict[Tuple, EdgeTensorType] = {}
        self._edge_attr: Dict[Tuple, EdgeAttr] = {}
        self._edge_id: Dict[Tuple, Tensor] = {}

        self.num_partitions: int = 1
        self.partition_idx: int = 0
        # node_partition_book:  mapping between node ids and partition idx
        self.node_pb: Union[Tensor, Dict[NodeType, Tensor]] = None
        # edge_partition_book:  mapping between edge ids and partition idx
        self.edge_pb: Union[Tensor, Dict[EdgeType, Tensor]] = None
        # meta information related to partition and graph store info
        self.meta: Optional[Dict[Any, Any]] = None
        # partition labels
        self.labels: Union[Tensor, Dict[EdgeType, Tensor]] = None

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value)

    def get_partition_ids_from_nids(self, ids: torch.Tensor,
                            node_type: Optional[NodeType]=None):
        # Get the local partition ids of node ids with a specific node type.
        if self.meta["is_hetero"]:
            assert node_type is not None
            return self.node_pb[node_type][ids]
        return self.node_pb[ids]

    def get_partition_ids_from_eids(self, eids: torch.Tensor,
                            edge_type: Optional[EdgeType]=None):
        r""" Get the partition ids of edge ids with a specific edge type."""
        if self.meta["is_hetero"]:
            assert edge_type is not None
            return self.edge_pb[edge_type][eids]
        return self.edge_pb[eids]



    # starting for graph ..

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
    ) -> 'LocalGraphStore':
        r"""Creates a local graph store from a homogeneous :pyg:`PyG` graph.

        Args:
            edge_id (torch.Tensor): The global identifier for every local edge.
            edge_index (torch.Tensor): The local edge indices.
            num_nodes (int): The number of nodes in the local graph.
        """
        attr = dict(
            edge_type=None,
            layout='coo',
            size=(num_nodes, num_nodes),
        )

        graph_store = cls()
        graph_store.put_edge_index(edge_index, **attr)
        graph_store.put_edge_id(edge_id, **attr)
        return graph_store

    @classmethod
    def from_hetero_data(
        cls,
        edge_id_dict: Dict[EdgeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
    ) -> 'LocalGraphStore':
        r"""Creates a local graph store from a heterogeneous :pyg:`PyG` graph.

        Args:
            edge_id_dict (Dict[EdgeType, torch.Tensor]): The global identifier
                for every local edge of every edge type.
            edge_index_dict (Dict[EdgeType, torch.Tensor]): The local edge
                indices of every edge type.
            num_nodes_dict (Dict[NodeType, int]): The number of nodes in the
                local graph of every node type.
        """
        attr_dict = {}
        for edge_type in edge_index_dict.keys():
            src, _, dst = edge_type
            attr_dict[edge_type] = dict(
                edge_type=edge_type,
                layout='coo',
                size=(num_nodes_dict[src], num_nodes_dict[dst]),
            )

        graph_store = cls()
        for edge_type, edge_index in edge_index_dict.items():
            attr = attr_dict[edge_type]
            edge_id = edge_id_dict[edge_type]
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

        if not meta['is_hetero']:
            attr = dict(edge_type=None, layout='coo', size=graph_data['size'])
            graph_store.put_edge_index(torch.stack((graph_data['row'], graph_data['col']),dim=0),
                                       **attr)
            graph_store.put_edge_id(graph_data['edge_id'], **attr)

        if meta['is_hetero']:
            for edge_type, data in graph_data.items():
                attr = dict(edge_type=edge_type, layout='coo',
                            size=data['size'])
                graph_store.put_edge_index(torch.stack((data['row'], data['col']),dim=0),
                                       **attr)
                #graph_store.put_edge_index((data['row'], data['col']), **attr)
                graph_store.put_edge_id(data['edge_id'], **attr)

        return graph_store
