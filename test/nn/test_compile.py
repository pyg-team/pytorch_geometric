import torch

from torch_geometric.profile import benchmark

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)

    def gather_scatter(x, edge_index):
        row, col = edge_index
        x_j = x[row]
        col = col.view(-1, 1).expand(-1, x.size(-1))
        return torch.zeros_like(x).scatter_add_(0, col, x_j)

    benchmark(
        funcs=[gather_scatter, torch.compile(gather_scatter)],
        func_names=['Vanilla', 'Compiled'],
        args=(x, edge_index),
        num_steps=100 if args.device == 'cpu' else 1000,
        num_warmups=50 if args.device == 'cpu' else 500,
        backward=args.backward,
    )

    # # TODO Test aggregation package
    # # TODO Test conv package
    # # TODO Test different reductions
    # # TODO Test concat + transformation
    # # TODO Test dynamic
