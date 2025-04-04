import os.path as osp

import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GPSE
from torch_geometric.nn.models.gpse import (
    GPSENodeEncoder,
    gpse_loss,
    gpse_process,
    process_batch_idx,
)
from torch_geometric.testing import is_full_test
from torch_geometric.transforms import AddGPSE, VirtualNode

num_nodes = 6
size_node_targets = 51
size_graph_targets = 11
gpse_inner_dim = 512


def test_gpse_training():
    x = torch.randn(num_nodes, 20)
    y = torch.randn(num_nodes, size_node_targets)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])

    vn = VirtualNode()
    data = vn(Data(x=x, y=y, edge_index=edge_index))
    data.y_graph = torch.tensor(torch.randn(size_graph_targets))
    batch = Batch.from_data_list([data])
    model = GPSE()

    with torch.no_grad():
        out = model(batch)
        assert out[0].size() == out[1].size()
        assert out[0].size() == (num_nodes + 1,
                                 size_node_targets + size_graph_targets)

    if is_full_test():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        batch = Batch.from_data_list([data])

        min_loss = float('inf')
        for i in range(100):
            optimizer.zero_grad()
            pred, true = model(batch.clone())
            batch_idx = process_batch_idx(batch.batch, true)
            loss, _ = gpse_loss(pred, true, batch_idx)
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2


def test_gpse_from_pretrained():
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    vn = VirtualNode()
    data = vn(Data(x=x, edge_index=edge_index))

    model = GPSE.from_pretrained(
        name='molpcba', root=osp.join(osp.dirname(osp.realpath(__file__)),
                                      '..', 'data', 'GPSE_pretrained'))

    with torch.no_grad():
        out = gpse_process(model, data, 'NormalSE')
        assert out.size() == (num_nodes + 1, gpse_inner_dim)


def test_gpse_node_encoder():
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    model = GPSE.from_pretrained(
        name='molpcba', root=osp.join(osp.dirname(osp.realpath(__file__)),
                                      '..', 'data', 'GPSE_pretrained'))
    transform = AddGPSE(model)
    out = transform(data)
    encoder = GPSENodeEncoder(dim_emb=128, dim_pe_in=512, dim_pe_out=64,
                              expand_x=False)
    encoder_expand = GPSENodeEncoder(dim_emb=128, dim_pe_in=512, dim_pe_out=64,
                                     dim_in=4, expand_x=True)
    out, out_expand = encoder(out.clone()), encoder_expand(out.clone())
    assert out.pestat_GPSE.size() == (num_nodes, gpse_inner_dim)
    assert out.x.size() == (num_nodes, 68)
    assert out_expand.x.size() == (num_nodes, 128)
