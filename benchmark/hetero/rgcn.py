import argparse
import time
from collections import defaultdict

import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import MessagePassing

SEED = 12345

parser = argparse.ArgumentParser()
parser.add_argument('--num_node_types', type=int, default=10)
parser.add_argument('--num_edge_types', type=int, default=100)
parser.add_argument('--avg_num_nodes', type=int, default=100)
parser.add_argument('--avg_degree', type=int, default=10)
parser.add_argument('--channels', type=int, default=64)


class SequentialRGCN(MessagePassing):
    def __init__(self, num_edge_types, channels):
        super().__init__(aggr='sum')
        torch.manual_seed(SEED)
        self.weight = torch.randn(num_edge_types, channels, channels)

    def forward(self, x_dict, edge_index_dict):
        outs_dict = defaultdict(list)
        for i, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
            src, _, dst = edge_type
            out = self.propagate(edge_index, x=(x_dict[src], x_dict[dst]))
            out = out @ self.weight[i]
            outs_dict[dst].append(out)

        out_dict = {}
        for key in x_dict.keys():
            out_dict[key] = torch.stack(outs_dict[key], dim=0).sum(dim=0)

        return out_dict


class VerticalRGCN(MessagePassing):
    # https://arxiv.org/pdf/2107.10015.pdf
    def __init__(self, num_edge_types, channels):
        super().__init__(aggr='sum')
        torch.manual_seed(SEED)
        self.weight = torch.randn(num_edge_types, channels, channels)

    def forward(self, x, edge_index, edge_type):
        edge_index = edge_index.clone()
        edge_index[1] += x.size(0) * edge_type

        out = self.propagate(
            edge_index,
            x=x,
            size=(x.size(0), self.weight.size(0) * x.size(0)),
        )

        out = out.view(self.weight.size(0), x.size(0), x.size(1))
        out = out @ self.weight
        out = out.sum(dim=0)

        return out


class HorizontalRGCN(MessagePassing):
    # https://arxiv.org/pdf/2107.10015.pdf
    def __init__(self, num_edge_types, channels):
        super().__init__(aggr='sum')
        torch.manual_seed(SEED)
        self.weight = torch.randn(num_edge_types, channels, channels)

    def forward(self, x, edge_index, edge_type):
        edge_index = edge_index.clone()
        edge_index[0] += x.size(0) * edge_type

        out = x.view(1, x.size(0), x.size(1)) @ self.weight
        out = out.view(self.weight.size(0) * x.size(0), x.size(1))

        out = self.propagate(
            edge_index,
            x=out,
            size=(self.weight.size(0) * x.size(0), x.size(0)),
        )

        return out


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    seed_everything(SEED)
    dataset = FakeHeteroDataset(
        num_graphs=1,
        num_node_types=args.num_node_types,
        num_edge_types=args.num_edge_types,
        avg_num_nodes=args.avg_num_nodes,
        avg_degree=args.avg_degree,
    )
    hetero_data = dataset[0]
    for node_type in hetero_data.node_types:
        store = hetero_data[node_type]
        store.x = torch.randn(store.num_nodes, args.channels)
    homo_data = hetero_data.to_homogeneous()
    print(homo_data)

    conv = SequentialRGCN(args.num_edge_types, args.channels)
    t = time.perf_counter()
    out_dict = conv(hetero_data.x_dict, hetero_data.edge_index_dict)
    t = time.perf_counter() - t
    out1 = torch.cat([out for out in out_dict.values()])
    print(f'{conv}: {t}')
    print(out1[:5, :5])

    conv = VerticalRGCN(args.num_edge_types, args.channels)
    t = time.perf_counter()
    out2 = conv(homo_data.x, homo_data.edge_index, homo_data.edge_type)
    t = time.perf_counter() - t
    print(f'{conv}: {t}')
    print(out2[:5, :5])

    conv = HorizontalRGCN(args.num_edge_types, args.channels)
    t = time.perf_counter()
    out3 = conv(homo_data.x, homo_data.edge_index, homo_data.edge_type)
    t = time.perf_counter() - t
    print(f'{conv}: {t}')
    print(out3[:5, :5])
