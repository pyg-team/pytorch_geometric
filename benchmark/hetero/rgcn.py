import argparse
from collections import defaultdict

import torch

from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import MessagePassing

torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--num_node_types', type=int, default=10)
parser.add_argument('--num_edge_types', type=int, default=100)
parser.add_argument('--avg_num_nodes', type=int, default=100)
parser.add_argument('--avg_degree', type=int, default=10)
parser.add_argument('--channels', type=int, default=64)
args = parser.parse_args()

dataset = FakeHeteroDataset(
    num_graphs=1,
    num_node_types=args.num_node_types,
    num_edge_types=args.num_edge_types,
    avg_num_nodes=args.avg_num_nodes,
    avg_degree=args.avg_degree,
)

data = dataset[0]


class SequentialRGCN(MessagePassing):
    def __init__(self, num_edge_types, channels):
        super().__init__(aggr='sum')
        self.weight = torch.randn(num_edge_types, channels, channels)

    def forward(self, x_dict, edge_index_dict):
        outs_dict = defaultdict(list)
        for i, (edge_type, edge_index) in enumerate(edge_index_dict):
            src, _, dst = edge_type
            out = self.propagate(edge_index, x=(x_dict[src], x_dict[dst]))
            out = out @ self.weight[i]
            outs_dict[dst].append(out)

        out_dict = {}
        for key, outs in outs_dict.items():
            out_dict[key] = torch.stack(outs, dim=0).sum(dim=0)

        return out_dict


class VerticalRGCN(MessagePassing):
    # https://arxiv.org/pdf/2107.10015.pdf
    def __init__(self, num_edge_types, channels):
        super().__init__(aggr='sum')
        self.weight = torch.randn(num_edge_types, channels, channels)

    def forward(self, x_dict, edge_index_dict):
        pass
