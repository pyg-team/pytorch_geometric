import json
import logging
import os
import os.path as osp
from typing import List, Optional, Union

import torch

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType


def record_mapping(root: str, mapping: torch.Tensor, type: str,
                   type_name: Optional[Union[NodeType, EdgeType]] = None):
    r""" Save a partition book of graph nodes/edges to record which
    partition they belong to.
    """
    assert type in ["node", "edge"]
    fpath = osp.join(root, f'{type}_map.pt')
    if type_name is not None:
        path = osp.join(root, f'{type}_map')
        os.makedirs(path, exist_ok=True)
        if type == 'node':
            fpath = osp.join(path, f'{type_name}.pt')
        else:
            fpath = osp.join(path, f'{EdgeTypeStr(type_name)}.pt')
    torch.save(mapping, fpath)


def save_feature_tensor(feature_store: LocalFeatureStore, group_name: str,
                        attr_name: str, index: torch.Tensor,
                        feature: torch.Tensor, global_id: torch.tensor):
    assert (feature.shape[0] == global_id.shape[0])
    feature_store.put_tensor(feature, group_name=group_name,
                             attr_name=attr_name, index=index)
    feature_store.put_global_id(global_id, group_name=group_name)


def store_single_feature(root: str, group_name: str, attr_name: str,
                         feature: torch.Tensor, global_id: torch.Tensor,
                         index: torch.Tensor, type: str):
    if feature is not None:
        assert type in ['node', 'edge']
        logging.info(f"save {type} feature")
        feature_store = LocalFeatureStore()
        save_feature_tensor(feature_store, group_name=group_name,
                            attr_name=attr_name, index=index, feature=feature,
                            global_id=global_id)
        torch.save(feature_store, osp.join(root, f'{type}_feats.pt'))


class Partitioner:
    r"""Partition the graph structure and its features of a
    :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object.
    Partitioned data output will be structured like this:

    **Homogeneous graphs:**

    .. code-block::

        root/
        |-- META.json
        |-- node_map.pt
        |-- edge_map.pt
        |-- part0/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt
        |-- part1/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt

    **Heterogeneous graphs:**

    .. code-block::

        root/
        |-- META.json
        |-- node_map/
            |-- ntype1.pt
            |-- ntype2.pt
        |-- edge_map/
            |-- etype1.pt
            |-- etype2.pt
        |-- part0/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt
        |-- part1/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt

    Args:
        data (Data or HeteroData): The data object.
        num_parts (int): The number of partitions.
        root (str): Root directory where the partitioned dataset should be
            saved.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        root: str,
    ):
        assert num_parts > 1

        self.data = data
        self.num_parts = num_parts
        self.root = root

    @property
    def is_hetero(self) -> bool:
        return isinstance(self.data, HeteroData)

    @property
    def node_types(self) -> List[NodeType]:
        return self.data.node_types if self.is_hetero else None

    @property
    def edge_types(self) -> List[EdgeType]:
        return self.data.edge_types if self.is_hetero else None

    def generate_partition(self):
        r"""Generates the partition."""
        os.makedirs(self.root, exist_ok=True)

        logging.info('Saving metadata')
        meta = {
            'num_parts': self.num_parts,
            'is_hetero': self.is_hetero,
            'node_types': self.node_types,
            'edge_types': self.node_types,
        }
        with open(osp.join(self.root, 'META.json'), 'w') as f:
            json.dump(meta, f)

        data = self.data.to_homogeneous() if self.is_hetero else self.data
        cluster_data = ClusterData(
            data,
            num_parts=self.num_parts,
            log=True,
            keep_inter_cluster_edges=True,
        )

        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        edge_perm = cluster_data.partition.edge_perm

        node_map = torch.empty(data.num_nodes, dtype=torch.int64)
        edge_map = torch.empty(data.num_edges, dtype=torch.int64)

        if self.is_hetero:
            edge_type_num = len(data._edge_type_names)
            node_type_num = len(data._node_type_names)
            eid_offset = 0
            edge_offset = {}
            nid_offset = 0
            node_offset = {}
            for node_name in data._node_type_names:
                node_offset[node_name] = nid_offset
                nid_offset = nid_offset + self.data.num_nodes_dict[node_name]
            for edge_name in data._edge_type_names:
                edge_offset[edge_name] = eid_offset
                eid_offset = eid_offset + self.data.num_edges_dict[edge_name]

            edge_start = 0
            for pid in range(self.num_parts):
                # save graph partition
                logging.info(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = int(partptr[pid])
                end_pos = int(partptr[pid + 1])
                edge_length = cluster_data[pid].num_edges
                part_edge_ids = edge_perm[edge_start:edge_start + edge_length]
                edge_map[part_edge_ids] = pid
                edge_start = edge_start + edge_length
                edge_type = cluster_data[pid].edge_type
                node_type = cluster_data[pid].node_type
                graph_store = LocalGraphStore()
                edge_feature_store = LocalFeatureStore()
                for etype_id in range(edge_type_num):
                    edge_name = data._edge_type_names[etype_id]
                    mask = (edge_type == etype_id)
                    local_row_ids = torch.masked_select(edge_index[0], mask)
                    local_col_ids = torch.masked_select(edge_index[1], mask)
                    global_row_ids = node_perm[local_row_ids + start_pos]
                    global_col_ids = node_perm[local_col_ids]
                    assert len(edge_name) == 3
                    src_name = edge_name[0]
                    dst_name = edge_name[2]
                    global_row_ids = global_row_ids - node_offset.get(src_name)
                    global_col_ids = global_col_ids - node_offset.get(dst_name)
                    type_edge_id = torch.masked_select(part_edge_ids, mask)
                    type_edge_id = type_edge_id - edge_offset.get(edge_name)

                    src_num = self.data.num_nodes_dict.get(src_name)
                    dst_num = self.data.num_nodes_dict.get(dst_name)
                    graph_store.put_edge_index(
                        edge_index=(global_row_ids, global_col_ids),
                        edge_type=edge_name, layout='coo',
                        size=(src_num, dst_num))
                    graph_store.put_edge_id(type_edge_id, edge_type=edge_name,
                                            layout='coo',
                                            size=(src_num, dst_num))
                    # save edge feature partition
                    if cluster_data[pid].edge_attr is not None:
                        logging.info(
                            f"save edge feature for edge type: {edge_name}")
                        type_edge_feat = cluster_data[pid].edge_attr[mask, :]
                        save_feature_tensor(edge_feature_store,
                                            group_name=edge_name,
                                            attr_name="edge_attr", index=None,
                                            feature=type_edge_feat,
                                            global_id=type_edge_id)

                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)
                torch.save(graph_store, osp.join(path, 'graph.pt'))
                if len(edge_feature_store.get_all_tensor_attrs()) > 0:
                    torch.save(edge_feature_store,
                               osp.join(path, 'edge_feats.pt'))

                # save node feature partition
                logging.info(f"save node feature for part: {pid}")
                node_ids = node_perm[start_pos:end_pos]
                node_map[node_ids] = pid
                if cluster_data[pid].x is not None:
                    node_feature_store = LocalFeatureStore()
                    for ntype_id in range(node_type_num):
                        node_name = data._node_type_names[ntype_id]
                        mask = (node_type == ntype_id)
                        type_node_id = torch.masked_select(node_ids, mask)
                        type_node_id = type_node_id - node_offset.get(
                            node_name)
                        type_node_feat = cluster_data[pid].x[mask, :]
                        save_feature_tensor(node_feature_store,
                                            group_name=node_name,
                                            attr_name='x', index=None,
                                            feature=type_node_feat,
                                            global_id=type_node_id)
                    torch.save(node_feature_store,
                               osp.join(path, 'node_feats.pt'))

            # save node partition mapping
            logging.info("save node partition mapping")
            for ntype_id in range(node_type_num):
                node_name = data._node_type_names[ntype_id]
                mask = (data.node_type == ntype_id)
                type_node_map = torch.masked_select(node_map, mask)
                record_mapping(self.root, type_node_map, 'node', node_name)

            # save edge partition mapping
            logging.info("save edge partition mapping")
            for etype_id in range(edge_type_num):
                edge_name = data._edge_type_names[etype_id]
                mask = (data.edge_type == etype_id)
                type_edge_map = torch.masked_select(edge_map, mask)
                record_mapping(self.root, type_edge_map, 'edge', edge_name)

        else:  # `if not self.is_hetero:`

            edge_offset = 0
            for pid in range(self.num_parts):
                logging.info(f'Saving graph partition {pid}')
                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)

                part_data = cluster_data[pid]
                start, end = int(partptr[pid]), int(partptr[pid + 1])

                num_edges = part_data.num_edges
                edge_id = edge_perm[edge_offset:edge_offset + num_edges]
                edge_map[edge_id] = pid
                edge_offset = edge_offset + num_edges

                node_id = node_perm[start:end]
                node_map[node_id] = pid

                torch.save(
                    {
                        'edge_id': edge_id,
                        'row': part_data.edge_index[0],
                        'col': part_data.edge_index[1],
                        'size': (data.num_nodes, data.num_nodes),
                    }, osp.join(path, 'graph.pt'))

                torch.save(
                    {
                        'global_id': node_id,
                        'feats': dict(x=part_data.x),
                    }, osp.join(path, 'node_feats.pt'))

                torch.save(
                    {
                        'global_id': edge_id,
                        'feats': dict(edge_attr=part_data.edge_attr),
                    }, osp.join(path, 'edge_feats.pt'))

            logging.info('Saving partition mapping info')
            torch.save(node_map, osp.join(self.root, 'node_map.pt'))
            torch.save(edge_map, osp.join(self.root, 'edge_map.pt'))
