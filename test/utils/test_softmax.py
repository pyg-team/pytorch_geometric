import torch

from torch_geometric.utils import softmax


def test_softmax():
    src = torch.tensor([1., 1., 1., 1.])
    index = torch.tensor([0, 0, 1, 2])
    ptr = torch.tensor([0, 2, 3, 4])

    out = softmax(src, index)
    assert out.tolist() == [0.5, 0.5, 1, 1]
    assert softmax(src, None, ptr).tolist() == out.tolist()

    src = src.view(-1, 1)
    out = softmax(src, index)
    assert out.tolist() == [[0.5], [0.5], [1], [1]]
    assert softmax(src, None, ptr).tolist() == out.tolist()

    jit = torch.jit.script(softmax)
    assert torch.allclose(jit(src, index), out)


def test_softmax_backward():
    src_sparse = torch.rand(4, 8)
    index = torch.tensor([0, 0, 1, 1])
    src_dense = src_sparse.clone().view(2, 2, src_sparse.size(-1))

    src_sparse.requires_grad_(True)
    src_dense.requires_grad_(True)

    out_sparse = softmax(src_sparse, index)
    out_sparse.mean().backward()
    out_dense = src_dense.softmax(dim=1)
    out_dense.mean().backward()

    assert torch.allclose(out_sparse, out_dense.view_as(out_sparse))
    assert torch.allclose(src_sparse.grad, src_dense.grad.view_as(src_sparse))


def test_softmax_dim():
    index = torch.tensor([0, 0, 0, 0])
    ptr = torch.tensor([0, 4])

    src = torch.randn(4)
    assert torch.allclose(softmax(src, index, dim=0), src.softmax(dim=0))
    assert torch.allclose(softmax(src, ptr=ptr, dim=0), src.softmax(dim=0))

    src = torch.randn(4, 16)
    assert torch.allclose(softmax(src, index, dim=0), src.softmax(dim=0))
    assert torch.allclose(softmax(src, ptr=ptr, dim=0), src.softmax(dim=0))

    src = torch.randn(4, 4)
    assert torch.allclose(softmax(src, index, dim=-1), src.softmax(dim=-1))
    assert torch.allclose(softmax(src, ptr=ptr, dim=-1), src.softmax(dim=-1))

    src = torch.randn(4, 4, 16)
    assert torch.allclose(softmax(src, index, dim=1), src.softmax(dim=1))
    assert torch.allclose(softmax(src, ptr=ptr, dim=1), src.softmax(dim=1))


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges, num_feats = 1000, 50000, 64

    num_warmups, num_steps = 500, 1000
    if args.device == 'cpu':
        num_warmups, num_steps = num_warmups // 10, num_steps // 10

    index = torch.randint(num_nodes - 5, (num_edges, ), device=args.device)
    out_grad = torch.randn(num_edges, num_feats, device=args.device)

    t_forward = t_backward = 0
    for i in range(num_warmups + num_steps):
        x = torch.randn(num_nodes, num_edges // num_nodes, num_feats,
                        device=args.device)
        if args.backward:
            x.requires_grad_(True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        out = x.softmax(dim=1)

        torch.cuda.synchronize()
        if i >= num_warmups:
            t_forward += time.perf_counter() - t_start

        if args.backward:
            t_start = time.perf_counter()
            out.backward(out_grad.view(num_nodes, -1, num_feats))

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_backward += time.perf_counter() - t_start

    print(f'Dense forward:   {t_forward:.4f}s')
    if args.backward:
        print(f'Dense backward:  {t_backward:.4f}s')
    print('========================')

    t_forward = t_backward = 0
    for i in range(num_warmups + num_steps):
        x = torch.randn(num_edges, num_feats, device=args.device)
        if args.backward:
            x.requires_grad_(True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        out = softmax(x, index)

        torch.cuda.synchronize()
        if i >= num_warmups:
            t_forward += time.perf_counter() - t_start

        if args.backward:
            t_start = time.perf_counter()
            out.backward(out_grad)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_backward += time.perf_counter() - t_start

    print(f'Sparse forward:  {t_forward:.4f}s')
    if args.backward:
        print(f'Sparse backward: {t_backward:.4f}s')
    print('========================')
