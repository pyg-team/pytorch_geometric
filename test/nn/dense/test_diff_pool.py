from itertools import product

import torch

from torch_geometric.nn import dense_diff_pool
from torch_geometric.profile import benchmark
from torch_geometric.testing import is_full_test


def test_dense_diff_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = torch.randn((batch_size, num_nodes, channels))
    adj = torch.rand((batch_size, num_nodes, num_nodes))
    s = torch.randn((batch_size, num_nodes, num_clusters))
    mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)

    x_out, adj_out, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
    assert x_out.size() == (2, 10, 16)
    assert adj_out.size() == (2, 10, 10)
    assert link_loss.item() >= 0
    assert ent_loss.item() >= 0

    if is_full_test():
        jit = torch.jit.script(dense_diff_pool)
        x_jit, adj_jit, link_loss, ent_loss = jit(x, adj, s, mask)
        assert torch.allclose(x_jit, x_out)
        assert torch.allclose(adj_jit, adj_out)
        assert link_loss.item() >= 0
        assert ent_loss.item() >= 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    BS = [2**i for i in range(4, 8)]
    NS = [2**i for i in range(4, 8)]
    FS = [2**i for i in range(5, 9)]
    CS = [2**i for i in range(5, 9)]

    funcs = []
    func_names = []
    args_list = []
    for B, N, F, C in product(BS, NS, FS, CS):
        x = torch.randn(B, N, F, device=args.device)
        adj = torch.randint(0, 2, (B, N, N), dtype=x.dtype, device=args.device)
        s = torch.randn(B, N, C, device=args.device)

        funcs.append(dense_diff_pool)
        func_names.append(f'B={B}, N={N}, F={F}, C={C}')
        args_list.append((x, adj, s))

    benchmark(
        funcs=funcs,
        func_names=func_names,
        args=args_list,
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        per_step=True,
        progress_bar=True,
    )
