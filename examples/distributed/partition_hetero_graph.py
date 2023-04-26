import argparse
import os.path as osp

from torch_geometric.datasets import OGB_MAG
from torch_geometric.partition.partitioner import Partitioner


def partition_dataset(ogbn_dataset: str, root_dir: str, num_partitions: int):
    save_dir = root_dir + "/partition"
    dataset = OGB_MAG(root=ogbn_dataset, preprocess='metapath2vec')
    data = dataset[0]

    partitioner = Partitioner(output_dir=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for ClusterData Partitioning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-mags',
        help="The name of dataset.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default='./data/mags',
        help="The root directory of input dataset and output partitions.",
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
