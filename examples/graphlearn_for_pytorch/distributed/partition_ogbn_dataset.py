# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import ast
import os.path as osp

import graphlearn_torch as glt
import torch
from ogb.nodeproppred import PygNodePropPredDataset


def partition_dataset(ogbn_dataset: str, root_dir: str, num_partitions: int,
                      num_nbrs: glt.NumNeighbors, chunk_size: int,
                      cache_ratio: float):
    print(f'-- Loading {ogbn_dataset} ...')
    dataset = PygNodePropPredDataset(ogbn_dataset, root_dir)
    data = dataset[0]
    node_num = len(data.x)
    edge_num = len(data.edge_index[0])
    print('* node count: {}'.format(node_num))
    print('* edge count: {}'.format(edge_num))
    split_idx = dataset.get_idx_split()

    print('-- Saving label ...')
    label_dir = osp.join(root_dir, f'{ogbn_dataset}-label')
    glt.utils.ensure_dir(label_dir)
    torch.save(data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training idx ...')
    train_idx = split_idx['train']
    train_idx = train_idx.split(train_idx.size(0) // num_partitions)
    train_idx_partitions_dir = osp.join(root_dir,
                                        f'{ogbn_dataset}-train-partitions')
    glt.utils.ensure_dir(train_idx_partitions_dir)
    for pidx in range(num_partitions):
        torch.save(train_idx[pidx],
                   osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

    print('-- Partitioning test idx ...')
    test_idx = split_idx['test']
    test_idx = test_idx.split(test_idx.size(0) // num_partitions)
    test_idx_partitions_dir = osp.join(root_dir,
                                       f'{ogbn_dataset}-test-partitions')
    glt.utils.ensure_dir(test_idx_partitions_dir)
    for pidx in range(num_partitions):
        torch.save(test_idx[pidx],
                   osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'))

    print('-- Initializing graph ...')
    csr_topo = glt.data.CSRTopo(edge_index=data.edge_index, layout='COO')
    graph = glt.data.Graph(csr_topo, mode='ZERO_COPY')

    print('-- Sampling hotness ...')
    glt_sampler = glt.sampler.NeighborSampler(graph, num_nbrs)
    node_probs = []
    for pidx in range(num_partitions):
        seeds = train_idx[pidx]
        prob = glt_sampler.sample_prob(seeds, node_num)
        node_probs.append(prob.cpu())

    print('-- Partitioning graph and features ...')
    partitions_dir = osp.join(root_dir, f'{ogbn_dataset}-partitions')
    freq_partitioner = glt.partition.FrequencyPartitioner(
        output_dir=partitions_dir, num_parts=num_partitions,
        num_nodes=node_num, edge_index=data.edge_index, probs=node_probs,
        node_feat=data.x, chunk_size=chunk_size, cache_ratio=cache_ratio)
    freq_partitioner.partition()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for partitioning ogbn datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-products',
        help="The name of ogbn dataset.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default='../../../data/ogbn-products',
        help=
        "The root directory (relative path) of input dataset and output partitions.",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=2,
        help="Number of partitions",
    )
    parser.add_argument(
        "--num_nbrs",
        type=ast.literal_eval,
        default='[15,10,5]',
        help="The number of neighbors to sample hotness for feature caching.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Chunk size for feature partitioning.",
    )
    parser.add_argument(
        "--cache_ratio",
        type=float,
        default=0.2,
        help="The proportion to cache features per partition.",
    )
    args = parser.parse_args()

    partition_dataset(
        ogbn_dataset=args.dataset,
        root_dir=osp.join(osp.dirname(osp.realpath(__file__)), args.root_dir),
        num_partitions=args.num_partitions, num_nbrs=args.num_nbrs,
        chunk_size=args.chunk_size, cache_ratio=args.cache_ratio)
