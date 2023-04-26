import json
import os
from typing import List, Optional, Union

import torch

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, as_str


def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_meta(
    output_dir: str,
    num_parts: int,
    data_cls: str = 'homo',
    node_types: Optional[List[NodeType]] = None,
    edge_types: Optional[List[EdgeType]] = None,
):
    r""" Save partitioning meta info into the output directory.
    """
    meta = {
        'num_parts': num_parts,
        'data_cls': data_cls,
        'node_types': node_types,
        'edge_types': edge_types
    }
    meta_file = os.path.join(output_dir, 'META.json')
    with open(meta_file, "w") as outfile:
        json.dump(meta, outfile, sort_keys=True, indent=4)


def save_node_pb(output_dir: str, node_pb: torch.Tensor,
                 ntype: Optional[NodeType] = None):
    r""" Save a partition book of graph nodes into the output directory.
    """
    if ntype is not None:
        subdir = os.path.join(output_dir, 'node_pb')
        ensure_dir(subdir)
        fpath = os.path.join(subdir, f'{as_str(ntype)}.pt')
    else:
        fpath = os.path.join(output_dir, 'node_pb.pt')
    torch.save(node_pb, fpath)


def save_edge_pb(output_dir: str, edge_pb: torch.Tensor,
                 etype: Optional[EdgeType] = None):
    r""" Save a partition book of graph edges into the output directory.
    """
    if etype is not None:
        subdir = os.path.join(output_dir, 'edge_pb')
        ensure_dir(subdir)
        fpath = os.path.join(subdir, f'{EdgeTypeStr(etype)}.pt')
    else:
        fpath = os.path.join(output_dir, 'edge_pb.pt')
    torch.save(edge_pb, fpath)


def save_graph_partition(output_dir: str, partition_idx: int,
                         row_ids: torch.Tensor, col_ids: torch.Tensor,
                         edge_ids: torch.Tensor,
                         etype: Optional[EdgeType] = None):
    r""" Save a graph topology partition into the output directory.
    """
    subdir = os.path.join(output_dir, f'part{partition_idx}', 'graph')
    if etype is not None:
        subdir = os.path.join(subdir, EdgeTypeStr(etype))
    ensure_dir(subdir)
    torch.save(row_ids, os.path.join(subdir, 'rows.pt'))
    torch.save(col_ids, os.path.join(subdir, 'cols.pt'))
    torch.save(edge_ids, os.path.join(subdir, 'eids.pt'))


def save_feature_partition(output_dir: str, partition_idx: int,
                           feature: torch.Tensor, ids: torch.Tensor,
                           group: str = 'node_feat',
                           type_name: Optional[Union[NodeType,
                                                     EdgeType]] = None):
    r""" Save a feature partition into the output directory.
    """
    subdir = os.path.join(output_dir, f'part{partition_idx}', group)
    if type_name is not None:
        if isinstance(type_name, NodeType):
            subdir = os.path.join(subdir, type_name)
        elif isinstance(type_name, EdgeType):
            subdir = os.path.join(subdir, EdgeTypeStr(type_name))

    ensure_dir(subdir)
    assert feature.shape[0] == ids.shape[0]
    torch.save(feature, os.path.join(subdir, 'feats.pt'))
    torch.save(ids, os.path.join(subdir, 'ids.pt'))


class Partitioner():
    r""" Base class for partitioning graphs and features.
  """
    def __init__(self, output_dir: str, num_parts: int,
                 data: Union[Data, HeteroData],
                 device: torch.device = torch.device('cpu')):

        self.output_dir = output_dir
        ensure_dir(self.output_dir)

        self.num_parts = num_parts
        assert self.num_parts > 1
        self.data = data
        if isinstance(data, HeteroData):
            self.data_cls = "hetero"
            self.node_types = data.node_types
            self.edge_types = data.edge_types
        else:
            self.data_cls = "homo"
            self.node_types = None
            self.edge_types = None
        self.device = device

    def generate_partition(self):
        r""" Partition graph and feature data into different parts.

    The output directory of partitioned graph data will be like:

    * homogeneous

      root_dir/
      |-- META
      |-- node_pb.pt
      |-- edge_pb.pt
      |-- part0/
          |-- graph/
              |-- rows.pt
              |-- cols.pt
              |-- eids.pt
          |-- node_feat/
              |-- feats.pt
              |-- ids.pt
              |-- cache_feats.pt (optional)
              |-- cache_ids.pt (optional)
          |-- edge_feat/
              |-- feats.pt
              |-- ids.pt
              |-- cache_feats.pt (optional)
              |-- cache_ids.pt (optional)
      |-- part1/
          |-- graph/
              ...
          |-- node_feat/
              ...
          |-- edge_feat/
              ...

    * heterogeneous

      root_dir/
      |-- META
      |-- node_pb/
          |-- ntype1.pt
          |-- ntype2.pt
      |-- edge_pb/
          |-- etype1.pt
          |-- etype2.pt
      |-- part0/
          |-- graph/
              |-- etype1/
                  |-- rows.pt
                  |-- cols.pt
                  |-- eids.pt
              |-- etype2/
                  ...
          |-- node_feat/
              |-- ntype1/
                  |-- feats.pt
                  |-- ids.pt
                  |-- cache_feats.pt (optional)
                  |-- cache_ids.pt (optional)
              |-- ntype2/
                  ...
          |-- edge_feat/
              |-- etype1/
                  |-- feats.pt
                  |-- ids.pt
                  |-- cache_feats.pt (optional)
                  |-- cache_ids.pt (optional)
              |-- etype2/
                  ...
      |-- part1/
          |-- graph/
              ...
          |-- node_feat/
              ...
          |-- edge_feat/
              ...

    """
        # save meta.
        print("save metadata for partition info")
        save_meta(self.output_dir, self.num_parts, self.data_cls,
                  self.node_types, self.edge_types)

        input_data = self.data
        if 'hetero' == self.data_cls:
            input_data = self.data.to_homogeneous()
        cluster_data = ClusterData(input_data, num_parts=self.num_parts,
                                   log=True, inter_cluster_edges=True)
        perm = cluster_data.perm
        partptr = cluster_data.partptr
        node_partition_book = torch.arange(input_data.num_nodes,
                                           dtype=torch.int64)
        edge_partition_book = torch.arange(input_data.num_edges,
                                           dtype=torch.int64)

        if 'hetero' == self.data_cls:
            edge_type_num = len(input_data._edge_type_names)
            node_type_num = len(input_data._node_type_names)

            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = partptr[pid]
                end_pos = partptr[pid + 1]
                part_edge_ids = cluster_data[pid].eid
                edge_type = cluster_data[pid].edge_type
                node_type = cluster_data[pid].node_type
                for etype_id in range(edge_type_num):
                    edge_name = input_data._edge_type_names[etype_id]
                    mask = (edge_type == etype_id)
                    local_row_ids = torch.masked_select(edge_index[0], mask)
                    local_col_ids = torch.masked_select(edge_index[1], mask)
                    global_row_ids = perm[local_row_ids + start_pos]
                    global_col_ids = perm[local_col_ids]
                    type_edge_ids = torch.masked_select(part_edge_ids, mask)
                    edge_partition_book[type_edge_ids] = pid
                    save_graph_partition(self.output_dir, pid, global_row_ids,
                                         global_col_ids, type_edge_ids,
                                         edge_name)
                    # save edge feature partition
                    if cluster_data[pid].edge_attr is not None:
                        print(f"save edge feature for edge type: {edge_name}")
                        type_edge_feat = cluster_data[pid].edge_attr[mask, :]
                        save_feature_partition(self.output_dir, pid,
                                               type_edge_feat, type_edge_ids,
                                               group="edge_feat",
                                               type_name=edge_name)

                # save node feature partition
                print(f"save node feature for part: {pid}")
                node_ids = perm[start_pos:end_pos]
                node_partition_book[node_ids] = pid
                if cluster_data[pid].x is not None:
                    offset = 0
                    for ntype_id in range(node_type_num):
                        node_name = input_data._node_type_names[ntype_id]
                        mask = (node_type == ntype_id)
                        type_node_id = torch.masked_select(node_ids, mask)
                        type_node_id = type_node_id - offset
                        offset = offset + self.data.num_nodes_dict[node_name]
                        type_node_feat = cluster_data[pid].x[mask, :]
                        save_feature_partition(self.output_dir, pid,
                                               type_node_feat, type_node_id,
                                               group="node_feat",
                                               type_name=node_name)

            # save node partition book
            print("save node partition book")
            for ntype_id in range(node_type_num):
                node_name = input_data._node_type_names[ntype_id]
                mask = (input_data.node_type == ntype_id)
                type_node_pb = torch.masked_select(node_partition_book, mask)
                save_node_pb(self.output_dir, type_node_pb, node_name)

            # save edge partition book
            print("save edge partition book")
            for etype_id in range(edge_type_num):
                edge_name = input_data._edge_type_names[etype_id]
                mask = (input_data.edge_type == etype_id)
                type_edge_pb = torch.masked_select(edge_partition_book, mask)
                save_edge_pb(self.output_dir, type_edge_pb, edge_name)

        else:
            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = partptr[pid]
                end_pos = partptr[pid + 1]
                local_row_ids = edge_index[0]
                local_col_ids = edge_index[1]
                global_row_ids = perm[local_row_ids + start_pos]
                global_col_ids = perm[local_col_ids]
                edge_ids = cluster_data[pid].eid
                save_graph_partition(self.output_dir, pid, global_row_ids,
                                     global_col_ids, edge_ids)
                edge_partition_book[edge_ids] = pid
                # save edge feature partition
                if cluster_data[pid].edge_attr is not None:
                    print(f"save edge feature for part: {pid}")
                    save_feature_partition(self.output_dir, pid,
                                           cluster_data[pid].edge_attr,
                                           edge_ids, group="edge_feat")

                # save node feature partition
                print(f"save node feature for part: {pid}")
                node_ids = perm[start_pos:end_pos]
                if cluster_data[pid].x is not None:
                    save_feature_partition(self.output_dir, pid,
                                           cluster_data[pid].x, node_ids,
                                           group="node_feat")

                node_partition_book[node_ids] = pid

            # save node/edge partition book
            print("save partition book for nodes/edges")
            save_node_pb(self.output_dir, node_partition_book, self.node_types)
            save_edge_pb(self.output_dir, edge_partition_book, self.edge_types)
