import json
import os
from typing import List, Optional, Union

import torch

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, as_str


def prepare_directory(root_path: str, child_path: Optional[str] = None):
    dir_path = root_path
    if child_path is not None:
        dir_path = os.path.join(root_path, child_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def record_meta_info(
    output_dir: str,
    num_parts: int,
    is_hetero: bool = False,
    node_types: Optional[List[NodeType]] = None,
    edge_types: Optional[List[EdgeType]] = None,
):
    r""" Save partitioning meta info into the output directory.
    """
    meta = {
        'num_parts': num_parts,
        'hetero_graph': is_hetero,
        'node_types': node_types,
        'edge_types': edge_types
    }
    meta_file = os.path.join(output_dir, 'META.json')
    with open(meta_file, "w") as outfile:
        json.dump(meta, outfile, sort_keys=True, indent=4)


def record_mapping(output_dir: str, mapping: torch.Tensor, type: str,
                   type_name: Optional[Union[NodeType, EdgeType]] = None):
    r""" Save a partition book of graph nodes/edges to record which
    partition they belong to.
    """
    assert type in ["node", "edge"]
    fpath = os.path.join(output_dir, f'{type}_map.pt')
    if type_name is not None:
        sub_dir = prepare_directory(output_dir, f'{type}_map')
        if type == 'node':
            fpath = os.path.join(sub_dir, f'{as_str(type_name)}.pt')
        else:
            fpath = os.path.join(sub_dir, f'{EdgeTypeStr(type_name)}.pt')
    torch.save(mapping, fpath)


class Partitioner():
    r""" partition graphs and features for homo/hetero graphs.
    Partitioned data output will be structured like this:

    * homo graph

      output_dir/
      |-- META.json
      |-- node_map.pt
      |-- edge_map.pt
      |-- part0/
          |-- graph.pt
          |-- features.pt
      |-- part1/
          |-- graph.pt
          |-- features.pt

    * hetero graph

      output_dir/
      |-- META.json
      |-- node_map/
          |-- ntype1.pt
          |-- ntype2.pt
      |-- edge_map/
          |-- etype1.pt
          |-- etype2.pt
      |-- part0/
          |-- graph.pt
          |-- features.pt
      |-- part1/
          |-- graph.pt
          |-- features.pt

    """
    def __init__(self, output_dir: str, num_parts: int,
                 data: Union[Data, HeteroData],
                 device: torch.device = torch.device('cpu')):

        self.output_dir = prepare_directory(output_dir)

        self.num_parts = num_parts
        assert self.num_parts > 1
        self.data = data
        self.is_hetero = False
        if isinstance(data, HeteroData):
            self.is_hetero = True
            self.node_types = data.node_types
            self.edge_types = data.edge_types
        else:
            self.node_types = None
            self.edge_types = None
        self.device = device

    def generate_partition(self):
        r""" Partition graph and feature data into different parts.
    """
        # save meta info for partition.
        print("save metadata for partition info")
        record_meta_info(self.output_dir, self.num_parts, self.is_hetero,
                         self.node_types, self.edge_types)

        input_data = self.data
        if self.is_hetero:
            input_data = self.data.to_homogeneous()
        cluster_data = ClusterData(input_data, num_parts=self.num_parts,
                                   log=True, keep_inter_cluster_edges=True)
        assert cluster_data.partition is not None
        perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        eids = cluster_data.partition.edge_perm
        node_partition_mapping = torch.arange(input_data.num_nodes,
                                              dtype=torch.int64)
        edge_partition_mapping = torch.arange(input_data.num_edges,
                                              dtype=torch.int64)

        if self.is_hetero:
            edge_type_num = len(input_data._edge_type_names)
            node_type_num = len(input_data._node_type_names)
            eid_offset = 0
            edge_offset = {}
            nid_offset = 0
            node_offset = {}
            for node_name in input_data._node_type_names:
                node_offset[node_name] = nid_offset
                nid_offset = nid_offset + self.data.num_nodes_dict[node_name]
            for edge_name in input_data._edge_type_names:
                edge_offset[edge_name] = eid_offset
                eid_offset = eid_offset + self.data.num_edges_dict[edge_name]

            edge_start = 0
            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = int(partptr[pid])
                end_pos = int(partptr[pid + 1])
                edge_length = cluster_data[pid].num_edges
                part_edge_ids = eids[edge_start:edge_start + edge_length]
                edge_partition_mapping[part_edge_ids] = pid
                edge_start = edge_start + edge_length
                edge_type = cluster_data[pid].edge_type
                node_type = cluster_data[pid].node_type
                graph_store = LocalGraphStore()
                edge_attr_dict = {}
                edge_id_dict = {}
                for etype_id in range(edge_type_num):
                    edge_name = input_data._edge_type_names[etype_id]
                    mask = (edge_type == etype_id)
                    local_row_ids = torch.masked_select(edge_index[0], mask)
                    local_col_ids = torch.masked_select(edge_index[1], mask)
                    global_row_ids = perm[local_row_ids + start_pos]
                    global_col_ids = perm[local_col_ids]
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
                        print(f"save edge feature for edge type: {edge_name}")
                        type_edge_feat = cluster_data[pid].edge_attr[mask, :]
                        edge_attr_dict[edge_name] = type_edge_feat
                        edge_id_dict[edge_name] = type_edge_id

                sub_dir = prepare_directory(self.output_dir, f'part_{pid}')
                torch.save(graph_store, os.path.join(sub_dir, 'graph.pt'))

                # save node feature partition
                print(f"save node feature for part: {pid}")
                node_ids = perm[start_pos:end_pos]
                node_partition_mapping[node_ids] = pid
                node_feat_dict = {}
                node_id_dict = {}
                if cluster_data[pid].x is not None:
                    for ntype_id in range(node_type_num):
                        node_name = input_data._node_type_names[ntype_id]
                        mask = (node_type == ntype_id)
                        type_node_id = torch.masked_select(node_ids, mask)
                        type_node_id = type_node_id - node_offset.get(
                            node_name)
                        type_node_feat = cluster_data[pid].x[mask, :]
                        node_feat_dict[node_name] = type_node_feat
                        node_id_dict[node_name] = type_node_id

                feature_store = LocalFeatureStore.from_hetero_data(
                    node_id_dict, x_dict=node_feat_dict,
                    edge_id_dict=edge_id_dict, edge_attr_dict=edge_attr_dict)
                torch.save(feature_store, os.path.join(sub_dir, 'features.pt'))

            # save node partition mapping
            print("save node partition mapping")
            for ntype_id in range(node_type_num):
                node_name = input_data._node_type_names[ntype_id]
                mask = (input_data.node_type == ntype_id)
                type_node_map = torch.masked_select(node_partition_mapping,
                                                    mask)
                record_mapping(self.output_dir, type_node_map, 'node',
                               node_name)

            # save edge partition mapping
            print("save edge partition mapping")
            for etype_id in range(edge_type_num):
                edge_name = input_data._edge_type_names[etype_id]
                mask = (input_data.edge_type == etype_id)
                type_edge_map = torch.masked_select(edge_partition_mapping,
                                                    mask)
                record_mapping(self.output_dir, type_edge_map, 'edge',
                               edge_name)

        else:  # homo graph
            eid_offset = 0
            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = int(partptr[pid])
                end_pos = int(partptr[pid + 1])
                local_row_ids = edge_index[0]
                local_col_ids = edge_index[1]
                global_row_ids = perm[local_row_ids + start_pos]
                global_col_ids = perm[local_col_ids]
                edge_num = cluster_data[pid].num_edges
                part_edge_ids = eids[eid_offset:eid_offset + edge_num]
                eid_offset = eid_offset + edge_num
                node_num = input_data.num_nodes
                graph_store = LocalGraphStore()
                graph_store.put_edge_index(
                    edge_index=(global_row_ids, global_col_ids),
                    edge_type=None, layout='coo', size=(node_num, node_num))
                graph_store.put_edge_id(part_edge_ids,
                                        edge_type=self.edge_types,
                                        layout='coo',
                                        size=(node_num, node_num))
                sub_dir = prepare_directory(self.output_dir, f'part_{pid}')
                torch.save(graph_store, os.path.join(sub_dir, 'graph.pt'))

                edge_partition_mapping[part_edge_ids] = pid
                node_ids = perm[start_pos:end_pos]
                # save node/edge feature partition
                feature_store = LocalFeatureStore.from_data(
                    node_id=node_ids, x=cluster_data[pid].x,
                    edge_id=part_edge_ids,
                    edge_attr=cluster_data[pid].edge_attr)
                torch.save(feature_store, os.path.join(sub_dir, 'features.pt'))
                node_partition_mapping[node_ids] = pid

            # save node/edge partition mapping info
            print("save partition mapping info for nodes/edges")
            record_mapping(self.output_dir, node_partition_mapping, 'node',
                           self.node_types)
            record_mapping(self.output_dir, edge_partition_mapping, 'edge',
                           self.edge_types)
