import json
import os
from typing import List, Optional, Union

import torch

from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType


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
        fpath = os.path.join(subdir, 'ntype.pt')
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
                           graph_type: Optional[Union[NodeType,
                                                      EdgeType]] = None):
    r""" Save a feature partition into the output directory.
    """
    subdir = os.path.join(output_dir, f'part{partition_idx}', group)
    if graph_type is not None:
        if isinstance(graph_type, NodeType):
            subdir = os.path.join(subdir, graph_type)
        elif isinstance(graph_type, EdgeType):
            subdir = os.path.join(subdir, EdgeTypeStr(graph_type))

    ensure_dir(subdir)
    torch.save(feature, os.path.join(subdir, 'feats.pt'))
    torch.save(ids, os.path.join(subdir, 'ids.pt'))


class PartitionerBase():
    r""" Base class for partitioning graphs and features.
  """
    def __init__(self, output_dir: str, num_parts: int,
                 data: torch.utils.data.Dataset,
                 device: torch.device = torch.device('cpu')):

        self.output_dir = output_dir
        ensure_dir(self.output_dir)

        self.num_parts = num_parts
        assert self.num_parts > 1
        self.data = data
        if isinstance(data.num_nodes, dict):
            self.data_cls = "hetero"
            self.node_types = list(data.num_nodes.keys())
            self.edge_types = list(self.edge_index.keys())
        else:
            self.data_cls = "homo"
            self.node_types = None
            self.edge_types = None
        self.device = device

    def get_edge_index(self, etype: Optional[EdgeType] = None):
        if 'hetero' == self.data_cls:
            assert etype is not None
            return self.edge_index[etype]
        return self.edge_index

    def get_node_feat(self, ntype: Optional[NodeType] = None):
        if self.node_feat is None:
            return None
        if 'hetero' == self.data_cls:
            assert ntype is not None
            return self.node_feat[ntype]
        return self.node_feat

    def get_edge_feat(self, etype: Optional[EdgeType] = None):
        if self.edge_feat is None:
            return None
        if 'hetero' == self.data_cls:
            assert etype is not None
            return self.edge_feat[etype]
        return self.edge_feat

    def partition(self):
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
        if 'hetero' == self.data_cls:
            # TODO: add hetero support
            pass
        else:
            cluster_data = ClusterData(self.data, num_parts=self.num_parts,
                                       log=True, inter_cluster_edges=True)
            node_partition_book = torch.zeros(self.data.num_nodes,
                                              dtype=torch.long)
            edge_partition_book = torch.zeros(self.data.num_edges,
                                              dtype=torch.long)
            perm = cluster_data.perm
            partptr = cluster_data.partptr

            for pid in range(self.num_parts):
                # save graph partition
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
                # save edge partition book
                edge_partition_book[edge_ids] = pid
                # save edge feature partition
                if cluster_data[pid].edge_attr is not None:
                    save_feature_partition(self.output_dir, pid,
                                           cluster_data[pid].edge_attr,
                                           edge_ids, group="edge_feat")

                # save node feature partition
                node_ids = perm[start_pos:end_pos]
                if cluster_data[pid].x is not None:
                    save_feature_partition(self.output_dir, pid,
                                           cluster_data[pid].x, node_ids,
                                           group="node_feat")

                node_partition_book[node_ids] = pid

            # save node/edge partition book
            save_node_pb(self.output_dir, node_partition_book, self.node_types)
            save_edge_pb(self.output_dir, edge_partition_book, self.edge_types)
            # save meta.
            save_meta(self.output_dir, self.num_parts, self.data_cls,
                      self.node_types, self.edge_types)
