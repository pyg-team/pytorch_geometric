import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType, EdgeType, NodeType, as_str


class LocalGraphStore(GraphStore):
    r"""This class implements the :class:`torch_geometric.data.GraphStore`
    interface to act as a local graph store for distributed training."""
    def __init__(self):
        super().__init__()
        self._edge_index: Dict[Tuple, EdgeTensorType] = {}
        self._edge_attr: Dict[Tuple, EdgeAttr] = {}
        self._edge_id: Dict[Tuple, Tensor] = {}

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value)

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

        edge_attrs1 = graph_store.get_all_edge_attrs()

        graph_store.put_edge_id(edge_id, **attr)

        edge_attrs2 = graph_store.get_all_edge_attrs()
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
    def from_partition(
        self, root_dir: str, partition_idx: int
    ) -> Tuple[Dict, int, int, 'LocalGraphStore', torch.Tensor, torch.Tensor]:

        # load the partition from partition .pt files
        with open(os.path.join(root_dir, 'META.json'), 'rb') as infile:
            meta = json.load(infile)
        num_partitions = meta['num_parts']
        assert partition_idx >= 0
        assert partition_idx < num_partitions
        partition_dir = os.path.join(root_dir, f'part_{partition_idx}')
        assert os.path.exists(partition_dir)
        graph_dir = os.path.join(partition_dir, 'graph.pt')

        if os.path.exists(graph_dir):
            graph_data = torch.load(graph_dir)
        else:
            raise ValueError("not found graph files")

        if meta['is_hetero'] == False:
            #homo
            node_pb = torch.load(os.path.join(root_dir, 'node_map.pt'))
            edge_pb = torch.load(os.path.join(root_dir, 'edge_map.pt'))

            # initialize graph
            edge_index = torch.tensor(
                [graph_data['row'].tolist(), graph_data['col'].tolist()])
            graph_store = self.from_data(graph_data['edge_id'], edge_index,
                                         num_nodes=graph_data['size'][0])

            return (meta, num_partitions, partition_idx, graph_store, node_pb,
                    edge_pb)

        else:
            #hetero
            node_pb_dict = {}
            node_pb_dir = os.path.join(root_dir, 'node_map')
            for ntype in meta['node_types']:
                node_pb_dict[ntype] = torch.load(
                    os.path.join(node_pb_dir, f'{as_str(ntype)}.pt'))

            edge_pb_dict = {}
            edge_pb_dir = os.path.join(root_dir, 'edge_map')
            for etype in meta['edge_types']:
                edge_pb_dict[tuple(etype)] = torch.load(
                    os.path.join(edge_pb_dir, f'{as_str(etype)}.pt'))

            # convert partition data into dict.
            edge_id_dict, edge_index_dict, num_nodes_dict = {}, {}, {}
            for etype in meta['edge_types']:
                edge_id_dict[tuple(etype)] = graph_data[tuple(
                    etype)]['edge_id']
                edge_index_dict[tuple(etype)] = torch.tensor([
                    graph_data[tuple(etype)]['row'].tolist(),
                    graph_data[tuple(etype)]['col'].tolist()
                ])
                num_nodes_dict[etype[0]] = graph_data[tuple(etype)]['size'][0]
                num_nodes_dict[etype[2]] = graph_data[tuple(etype)]['size'][1]

            # initialize graph
            graph_store = self.from_hetero_data(edge_id_dict, edge_index_dict,
                                                num_nodes_dict)

            return (meta, num_partitions, partition_idx, graph_store,
                    node_pb_dict, edge_pb_dict)
