import json
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

import torch_geometric.distributed as pyg_dist
from torch_geometric.data import Data, HeteroData
from torch_geometric.io import fs
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.sampler.utils import sort_csc
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType


class Partitioner:
    r"""Partitions the graph and its features of a
    :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object.

    Partitioned data output will be structured as shown below.

    **Homogeneous graphs:**

    .. code-block:: none

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

    .. code-block:: none

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
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        root (str): Root directory where the partitioned dataset should be
            saved.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        root: str,
        recursive: bool = False,
    ):
        assert num_parts > 1

        self.data = data
        self.num_parts = num_parts
        self.root = root
        self.recursive = recursive

    @property
    def is_hetero(self) -> bool:
        return isinstance(self.data, HeteroData)

    @property
    def is_node_level_time(self) -> bool:
        if 'time' not in self.data:
            return False

        if self.is_hetero:
            return any(['time' in store for store in self.data.node_stores])

        return self.data.is_node_attr('time')

    @property
    def is_edge_level_time(self) -> bool:
        if 'edge_time' in self.data:
            return True

        if 'time' not in self.data:
            return False

        if self.is_hetero:
            return any(['time' in store for store in self.data.edge_stores])

        return self.data.is_edge_attr('time')

    @property
    def node_types(self) -> Optional[List[NodeType]]:
        return self.data.node_types if self.is_hetero else None

    @property
    def edge_types(self) -> Optional[List[EdgeType]]:
        return self.data.edge_types if self.is_hetero else None

    def generate_partition(self):
        r"""Generates the partitions."""
        os.makedirs(self.root, exist_ok=True)

        if self.is_hetero and self.is_node_level_time:
            time_data = {  # Get temporal information before converting data:
                node_type: self.data[node_type].time
                for node_type in self.data.node_types
            }

        data = self.data.to_homogeneous() if self.is_hetero else self.data
        cluster_data = ClusterData(
            data,
            num_parts=self.num_parts,
            recursive=self.recursive,
            log=True,
            keep_inter_cluster_edges=True,
            sparse_format='csc',
        )

        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        edge_perm = cluster_data.partition.edge_perm

        node_map = torch.empty(data.num_nodes, dtype=torch.int64)
        edge_map = torch.empty(data.num_edges, dtype=torch.int64)
        node_offset, edge_offset = {}, {}

        if self.is_hetero:
            offset = 0
            for node_type in self.node_types:
                node_offset[node_type] = offset
                offset += self.data[node_type].num_nodes

            offset = 0
            for edge_name in self.edge_types:
                edge_offset[edge_name] = offset
                offset += self.data.num_edges_dict[edge_name]

            edge_start = 0
            for pid in range(self.num_parts):
                logging.info(f'Saving graph partition {pid}')
                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)

                part_data = cluster_data[pid]
                start, end = int(partptr[pid]), int(partptr[pid + 1])

                num_edges = part_data.num_edges
                edge_id = edge_perm[edge_start:edge_start + num_edges]
                edge_map[edge_id] = pid
                edge_start += num_edges

                node_id = node_perm[start:end]
                node_map[node_id] = pid

                graph = {}
                efeat = defaultdict(dict)
                for i, edge_type in enumerate(self.edge_types):
                    # Row vector refers to source nodes.
                    # Column vector refers to destination nodes.
                    src, _, dst = edge_type
                    size = (self.data[src].num_nodes, self.data[dst].num_nodes)

                    mask = part_data.edge_type == i
                    row = part_data.edge_index[0, mask]
                    col = part_data.edge_index[1, mask]
                    global_col = node_id[col]
                    global_row = node_perm[row]

                    edge_time = src_node_time = None
                    if self.is_edge_level_time:
                        if 'edge_time' in part_data:
                            edge_time = part_data.edge_time[mask]
                        elif 'time' in part_data:
                            edge_time = part_data.time[mask]

                    elif self.is_node_level_time:
                        src_node_time = time_data[src]

                    offsetted_row = global_row - node_offset[src]
                    offsetted_col = global_col - node_offset[dst]
                    # Sort by column to avoid keeping track of permutations in
                    # `NeighborSampler` when converting to CSC format:
                    offsetted_row, offsetted_col, perm = sort_csc(
                        offsetted_row, offsetted_col, src_node_time, edge_time)

                    global_eid = edge_id[mask][perm]
                    assert torch.equal(
                        data.edge_index[:, global_eid],
                        torch.stack((offsetted_row + node_offset[src],
                                     offsetted_col + node_offset[dst]), dim=0),
                    )
                    offsetted_eid = global_eid - edge_offset[edge_type]
                    assert torch.equal(
                        self.data[edge_type].edge_index[:, offsetted_eid],
                        torch.stack((
                            offsetted_row,
                            offsetted_col,
                        ), dim=0),
                    )
                    graph[edge_type] = {
                        'edge_id': global_eid,
                        'row': offsetted_row,
                        'col': offsetted_col,
                        'size': size,
                    }

                    if 'edge_attr' in part_data:
                        edge_attr = part_data.edge_attr[mask][perm]
                        efeat[edge_type].update({
                            'global_id':
                            offsetted_eid,
                            'feats':
                            dict(edge_attr=edge_attr),
                        })
                    if self.is_edge_level_time:
                        efeat[edge_type].update({'edge_time': edge_time[perm]})

                torch.save(efeat, osp.join(path, 'edge_feats.pt'))
                torch.save(graph, osp.join(path, 'graph.pt'))

                nfeat = {}
                for i, node_type in enumerate(self.node_types):
                    mask = part_data.node_type == i
                    x = part_data.x[mask] if 'x' in part_data else None
                    nfeat[node_type] = {
                        'global_id': node_id[mask],
                        'id': node_id[mask] - node_offset[node_type],
                        'feats': dict(x=x),
                    }
                    if self.is_node_level_time:
                        nfeat[node_type].update({'time': time_data[node_type]})

                torch.save(nfeat, osp.join(path, 'node_feats.pt'))

            logging.info('Saving partition mapping info')
            path = osp.join(self.root, 'node_map')
            os.makedirs(path, exist_ok=True)
            for i, node_type in enumerate(self.node_types):
                mask = data.node_type == i
                torch.save(node_map[mask], osp.join(path, f'{node_type}.pt'))

            path = osp.join(self.root, 'edge_map')
            os.makedirs(path, exist_ok=True)
            for i, edge_type in enumerate(self.edge_types):
                mask = data.edge_type == i
                torch.save(
                    edge_map[mask],
                    osp.join(path, f'{EdgeTypeStr(edge_type)}.pt'),
                )

        else:  # `if not self.is_hetero:`
            edge_start = 0
            for pid in range(self.num_parts):
                logging.info(f'Saving graph partition {pid}')
                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)

                part_data = cluster_data[pid]
                start, end = int(partptr[pid]), int(partptr[pid + 1])

                num_edges = part_data.num_edges
                edge_id = edge_perm[edge_start:edge_start + num_edges]
                edge_map[edge_id] = pid
                edge_start += num_edges

                node_id = node_perm[start:end]  # global node_ids
                node_map[node_id] = pid  # 0 or 1

                row = part_data.edge_index[0]
                col = part_data.edge_index[1]

                global_col = node_id[col]  # part_ids -> global
                global_row = node_perm[row]

                edge_time = node_time = None
                if self.is_edge_level_time:
                    if 'edge_time' in part_data:
                        edge_time = part_data.edge_time
                    elif 'time' in part_data:
                        edge_time = part_data.time

                elif self.is_node_level_time:
                    node_time = data.time

                # Sort by column to avoid keeping track of permuations in
                # `NeighborSampler` when converting to CSC format:
                global_row, global_col, perm = sort_csc(
                    global_row, global_col, node_time, edge_time)

                edge_id = edge_id[perm]

                assert torch.equal(
                    self.data.edge_index[:, edge_id],
                    torch.stack((global_row, global_col)),
                )
                if 'edge_attr' in part_data:
                    edge_attr = part_data.edge_attr[perm]
                    assert torch.equal(self.data.edge_attr[edge_id, :],
                                       edge_attr)

                torch.save(
                    {
                        'edge_id': edge_id,
                        'row': global_row,
                        'col': global_col,
                        'size': (data.num_nodes, data.num_nodes),
                    }, osp.join(path, 'graph.pt'))

                nfeat = {
                    'global_id': node_id,
                    'feats': dict(x=part_data.x),
                }
                if self.is_node_level_time:
                    nfeat.update({'time': data.time})

                torch.save(nfeat, osp.join(path, 'node_feats.pt'))

                efeat = defaultdict()
                if 'edge_attr' in part_data:
                    efeat.update({
                        'global_id':
                        edge_id,
                        'feats':
                        dict(edge_attr=part_data.edge_attr[perm]),
                    })
                if self.is_edge_level_time:
                    efeat.update({'edge_time': edge_time[perm]})

                torch.save(efeat, osp.join(path, 'edge_feats.pt'))

            logging.info('Saving partition mapping info')
            torch.save(node_map, osp.join(self.root, 'node_map.pt'))
            torch.save(edge_map, osp.join(self.root, 'edge_map.pt'))

        logging.info('Saving metadata')
        meta = {
            'num_parts': self.num_parts,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'node_offset': list(node_offset.values()) if node_offset else None,
            'is_hetero': self.is_hetero,
            'is_sorted': True,  # Based on colum/destination.
        }
        with open(osp.join(self.root, 'META.json'), 'w') as f:
            json.dump(meta, f)


def load_partition_info(
    root_dir: str,
    partition_idx: int,
) -> Tuple[Dict, int, int, torch.Tensor, torch.Tensor]:
    # load the partition with PyG format (graphstore/featurestore)
    with open(osp.join(root_dir, 'META.json'), 'rb') as infile:
        meta = json.load(infile)
    num_partitions = meta['num_parts']
    assert partition_idx >= 0
    assert partition_idx < num_partitions
    partition_dir = osp.join(root_dir, f'part_{partition_idx}')
    assert osp.exists(partition_dir)

    if meta['is_hetero'] is False:
        node_pb = fs.torch_load(osp.join(root_dir, 'node_map.pt'))
        edge_pb = fs.torch_load(osp.join(root_dir, 'edge_map.pt'))

        return (meta, num_partitions, partition_idx, node_pb, edge_pb)
    else:
        node_pb_dict = {}
        node_pb_dir = osp.join(root_dir, 'node_map')
        for ntype in meta['node_types']:
            node_pb_dict[ntype] = fs.torch_load(
                osp.join(node_pb_dir, f'{pyg_dist.utils.as_str(ntype)}.pt'))

        edge_pb_dict = {}
        edge_pb_dir = osp.join(root_dir, 'edge_map')
        for etype in meta['edge_types']:
            edge_pb_dict[tuple(etype)] = fs.torch_load(
                osp.join(edge_pb_dir, f'{pyg_dist.utils.as_str(etype)}.pt'))

        return (meta, num_partitions, partition_idx, node_pb_dict,
                edge_pb_dict)
