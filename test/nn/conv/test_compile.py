import warnings

import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import withPackage

x = torch.randn(4, 16)
edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
row, col = edge_index
value = torch.rand(row.size(0))
adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
adj1 = adj2.set_value(None)
adj3 = adj1.to_torch_sparse_coo_tensor()
adj4 = adj2.to_torch_sparse_coo_tensor()


@withPackage('torch==2.0.0')
def test_gcn_compile():
    conv = GCNConv(16, 32)
    compile_conv = torch.compile(conv)
    out1 = conv(x, edge_index)
    assert torch.allclose(compile_conv(x, edge_index), out1, atol=1e-6)
    # TODO(fix): This fails
    # assert torch.allclose(compile_conv(x, adj3.t()), out1, atol=1e-6)


@withPackage('torch==2.0.0')
def test_sage_compile():
    conv = SAGEConv(16, 32)
    compile_conv = torch.compile(conv)
    out1 = conv(x, edge_index)
    assert torch.allclose(compile_conv(x, edge_index), out1, atol=1e-6)
    # TODO(fix): This fails
    # assert torch.allclose(compile_conv(x, adj3.t()), out1, atol=1e-6)


if __name__ == '__main__':
    import argparse

    warnings.filterwarnings('ignore', ".*via the 'torch-scatter' package.*")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    convs = [GCNConv(16, 32), SAGEConv(16, 32)]
    for conv in convs:
        conv_name = conv.__class__.__name__
        benchmark(
            funcs=[
                conv,
                torch.compile(conv),
            ],
            func_names=[conv_name, 'Compiled' + conv_name],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
