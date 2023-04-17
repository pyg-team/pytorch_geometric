import argparse
import os
import os.path as osp
import pickle

import torch
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import ClusterData


def partition_dataset(ogbn_dataset: str, root_dir: str, num_partitions: int):
    save_dir = root_dir + "/partition"
    dataset = PygNodePropPredDataset(ogbn_dataset, root_dir)
    data = dataset[0]

    cluster_data = ClusterData(data, num_parts=num_partitions, log=True,
                               inter_cluster_edges=True)

    node_partition_book = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_partition_book = torch.zeros(data.num_edges, dtype=torch.long)
    perm = cluster_data.perm

    for pid in range(num_partitions):
        start_pos = cluster_data.partptr[pid]
        end_pos = cluster_data.partptr[pid + 1]
        # save graph partition
        graph_subdir = os.path.join(save_dir, f'part{pid}', 'graph')
        if not os.path.exists(graph_subdir):
            os.makedirs(graph_subdir)
        edge_index = cluster_data[pid].edge_index
        local_row_ids = edge_index[0]
        local_col_ids = edge_index[1]
        global_row_ids = perm[local_row_ids + start_pos]
        global_col_ids = perm[local_col_ids]
        torch.save(global_row_ids, os.path.join(graph_subdir, 'rows.pt'))
        torch.save(global_col_ids, os.path.join(graph_subdir, 'cols.pt'))
        edge_ids = cluster_data[pid].eid
        torch.save(edge_ids, os.path.join(graph_subdir, 'eids.pt'))
        edge_partition_book[edge_ids] = pid

        # save edge feature partition
        if cluster_data[pid].edge_attr != None:
            edge_feature_subdir = os.path.join(save_dir, f'part{pid}',
                                               'edge_feat')
            if not os.path.exists(edge_feature_subdir):
                os.makedirs(edge_feature_subdir)
            torch.save(cluster_data[pid].edge_attr,
                       os.path.join(edge_feature_subdir, 'feats.pt'))

        # save node feature partition
        node_feature_subdir = os.path.join(save_dir, f'part{pid}', 'node_feat')
        if not os.path.exists(node_feature_subdir):
            os.makedirs(node_feature_subdir)
        if cluster_data[pid].x != None:
            torch.save(cluster_data[pid].x,
                       os.path.join(node_feature_subdir, 'feats.pt'))

        node_partition_book[perm[start_pos:end_pos]] = pid
        torch.save(perm[start_pos:end_pos],
                   os.path.join(node_feature_subdir, 'ids.pt'))

    # save node/edge partition book
    torch.save(edge_partition_book, save_dir + "/edge_pb.pt")
    torch.save(node_partition_book, save_dir + "/node_pb.pt")

    # save meta info for graph partitions
    meta = {
        'num_parts': num_partitions,
        'data_cls': "homo",
        'node_types': None,
        'edge_types': None
    }
    with open(os.path.join(save_dir, 'META'), 'wb') as outfile:
        pickle.dump(meta, outfile, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for ClusterData Partitioning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-products',
        help="The name of dataset.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default='./data/products',
        help=
        "The root directory (relative path) of input dataset and output partitions.",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=2,
        help="Number of partitions",
    )
    args = parser.parse_args()
    partition_dataset(
        ogbn_dataset=args.dataset,
        root_dir=osp.join(osp.dirname(osp.realpath(__file__)),
                          args.root_dir), num_partitions=args.num_partitions)
