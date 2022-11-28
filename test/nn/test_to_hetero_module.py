import pytest  # noqa
import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import ToHeteroModule
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.dense import Linear

heterodata = HeteroData()
heterodata['v0'].x = torch.randn(20, 10)
heterodata['v1'].x = torch.randn(20, 10)
heterodata[('v0', 'r0',
            'v0')].edge_index = torch.randint(high=20,
                                              size=(2, 30)).to(torch.long)
heterodata[('v0', 'r2',
            'v1')].edge_index = torch.randint(high=20,
                                              size=(2, 30)).to(torch.long)
heterodata[('v1', 'r3',
            'v0')].edge_index = torch.randint(high=20,
                                              size=(2, 30)).to(torch.long)
heterodata[('v1', 'r4', 'v1')] = torch.randn(2, 50)


def test_to_hetero_linear():
    lin = Linear(10, 5)
    heterolin = ToHeteroModule(lin, heterodata.metadata())
    # test dict input
    x_dict = heterodata.collect('x')
    out = heterolin(x_dict)
    assert out['v0'].shape == (20, 5)
    assert out['v1'].shape == (20, 5)
    # test fused input
    x = torch.cat([x_j for x_j in x_dict.values()])
    node_type = torch.cat([(j * torch.ones(x_j.shape[0])).long()
                           for j, x_j in enumerate(x_dict.values())])
    out = heterolin(x=x, node_type=node_type)
    assert out.shape == (40, 5)


def test_to_hetero_gcn():
    gcnconv = GCNConv(10, 5)
    rgcnconv = ToHeteroModule(gcnconv, heterodata.metadata())
    # test dict input
    x_dict = heterodata.collect('x')
    e_idx_dict = heterodata.collect('edge_index')
    out = rgcnconv(x_dict, edge_index=e_idx_dict)
    assert out['v0'].shape == (20, 5)
    assert out['v1'].shape == (20, 5)

    x = torch.cat(list(x_dict.values()), dim=0)

    num_node_dict = heterodata.collect('num_nodes')
    increment_dict = {}
    ctr = 0
    for node_type in num_node_dict:
        increment_dict[node_type] = ctr
        ctr += num_node_dict[node_type]

    etypes_list = []
    for i, e_type in enumerate(e_idx_dict.keys()):
        src_type, dst_type = e_type[0], e_type[-1]
        if torch.numel(e_idx_dict[e_type]) != 0:
            e_idx_dict[e_type][
                0, :] = e_idx_dict[e_type][0, :] + increment_dict[src_type]
            e_idx_dict[e_type][
                1, :] = e_idx_dict[e_type][1, :] + increment_dict[dst_type]
            etypes_list.append(torch.ones(e_idx_dict[e_type].shape[-1]) * i)
    edge_type = torch.cat(etypes_list).to(torch.long)
    edge_index = torch.cat(list(e_idx_dict.values()), dim=1)
    # test fused input
    out = rgcnconv(x, edge_index=edge_index, edge_type=edge_type)
    assert out.shape == (40, 5)
