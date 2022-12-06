import pytest
import torch

from torch_geometric.nn import HeteroLayerNorm, LayerNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(affine, mode):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = LayerNorm(16, affine=affine, mode=mode)
    assert norm.__repr__() == f'LayerNorm(16, mode={mode})'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_types, num_nodes, num_feats = 100, 2000, 64

    num_warmups, num_steps = 100, 200
    if args.device == 'cpu':
        num_warmups, num_steps = num_warmups // 10, num_steps // 10

    type_vec = torch.randint(0, num_types, (num_types * num_nodes, ),
                             device=args.device)
    batch = torch.randint(0, 1024, (num_types * num_nodes, ),
                          device=args.device)
    out_grad = torch.randn(num_types * num_nodes, num_feats,
                           device=args.device)

    for mode in ['node']:  # TODO mode='graph'
        print(f'Mode: {mode}')
        print('=========================')

        norm = LayerNorm(num_feats, mode=mode)
        norm.to(args.device)

        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            x = torch.randn(num_types * num_nodes, num_feats,
                            device=args.device)
            if args.backward:
                x.requires_grad_(True)
            xs = x.split(num_nodes, dim=0)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = torch.cat([norm(x) for x in xs], dim=0)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'Vanilla forward:  {t_forward:.4f}s')
        if args.backward:
            print(f'Vanilla backward: {t_backward:.4f}s')
        print('=========================')

        hetero_norm = HeteroLayerNorm(num_feats, num_types, mode=mode)
        hetero_norm.to(args.device)

        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            x = torch.randn(num_types * num_nodes, num_feats,
                            device=args.device)
            if args.backward:
                x.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = hetero_norm(x, type_vec)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'Fused forward:    {t_forward:.4f}s')
        if args.backward:
            print(f'Fused backward:   {t_backward:.4f}s')
        print('=========================')

        print()
