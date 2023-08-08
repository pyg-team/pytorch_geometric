import argparse
import ast
import os.path as osp

import graphlearn_torch as glt
import torch
from ogb.nodeproppred import PygNodePropPredDataset


def partition_dataset(
    ogbn_dataset: str,
    root_dir: str,
    num_partitions: int,
    num_nbrs: glt.NumNeighbors,
    chunk_size: int,
    cache_ratio: float,
):
    ###########################################################################
    # In distributed training (under the worker mode), each node in the cluster
    # holds a partition of the graph. Thus before the training starts, we
    # partition the dataset into multiple partitions, each of which corresponds
    # to a specific training worker.
    # The partitioning occurs in three steps:
    #   1. Run a partition algorithm to assign nodes to partitions.
    #   2. Construct partition graph structure based on the node assignment.
    #   3. Split the node features and edge features based on the partition
    # result.
    ###########################################################################

    print(f'-- Loading {ogbn_dataset} ...')
    dataset = PygNodePropPredDataset(ogbn_dataset, root_dir)
    data = dataset[0]
    print(f'* node count: {data.num_nodes}')
    print(f'* edge count: {data.num_edges}')
    split_idx = dataset.get_idx_split()

    print('-- Saving label ...')
    label_dir = osp.join(root_dir, f'{ogbn_dataset}-label')
    glt.utils.ensure_dir(label_dir)
    torch.save(data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training idx ...')
    train_idx = split_idx['train']
    train_idx = train_idx.split(train_idx.size(0) // num_partitions)
    train_idx_partitions_dir = osp.join(
        root_dir,
        f'{ogbn_dataset}-train-partitions',
    )
    glt.utils.ensure_dir(train_idx_partitions_dir)
    for pidx in range(num_partitions):
        torch.save(
            train_idx[pidx],
            osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'),
        )

    print('-- Partitioning test idx ...')
    test_idx = split_idx['test']
    test_idx = test_idx.split(test_idx.size(0) // num_partitions)
    test_idx_partitions_dir = osp.join(
        root_dir,
        f'{ogbn_dataset}-test-partitions',
    )
    glt.utils.ensure_dir(test_idx_partitions_dir)
    for pidx in range(num_partitions):
        torch.save(
            test_idx[pidx],
            osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'),
        )

    print('-- Initializing graph ...')
    csr_topo = glt.data.Topology(edge_index=data.edge_index,
                                 input_layout='COO')
    graph = glt.data.Graph(csr_topo, mode='ZERO_COPY')

    print('-- Sampling hotness ...')
    glt_sampler = glt.sampler.NeighborSampler(graph, num_nbrs)
    node_probs = []
    for pidx in range(num_partitions):
        seeds = train_idx[pidx]
        prob = glt_sampler.sample_prob(seeds, data.num_nodes)
        node_probs.append(prob.cpu())

    print('-- Partitioning graph and features ...')
    partitions_dir = osp.join(root_dir, f'{ogbn_dataset}-partitions')
    freq_partitioner = glt.partition.FrequencyPartitioner(
        output_dir=partitions_dir,
        num_parts=num_partitions,
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        probs=node_probs,
        node_feat=data.x,
        chunk_size=chunk_size,
        cache_ratio=cache_ratio,
    )
    freq_partitioner.partition()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-products',
        help='The name of the dataset',
    )
    parser.add_argument(
        '--num_partitions',
        type=int,
        default=2,
        help='The Number of partitions',
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='../../../data/ogbn-products',
        help='The root directory (relative path) of the partitioned dataset',
    )
    parser.add_argument(
        '--num_nbrs',
        type=ast.literal_eval,
        default='[15,10,5]',
        help='The number of neighbors to sample hotness for feature caching',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=10000,
        help='The chunk size for feature partitioning',
    )
    parser.add_argument(
        '--cache_ratio',
        type=float,
        default=0.2,
        help='The proportion to cache features per partition',
    )
    args = parser.parse_args()

    partition_dataset(
        ogbn_dataset=args.dataset,
        root_dir=osp.join(osp.dirname(osp.realpath(__file__)), args.root_dir),
        num_partitions=args.num_partitions,
        num_nbrs=args.num_nbrs,
        chunk_size=args.chunk_size,
        cache_ratio=args.cache_ratio,
    )
