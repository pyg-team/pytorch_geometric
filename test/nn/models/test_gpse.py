import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GPSE, GPSENodeEncoder
from torch_geometric.nn.models.gpse import (IdentityHead, gpse_loss,
                                            gpse_process, process_batch_idx)
from torch_geometric.testing import is_full_test
from torch_geometric.transforms import VirtualNode


def test_gpse_training():
    x = torch.randn(6, 20)
    y = torch.randn(6, 51)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])

    data = Data(x=x, y=y, edge_index=edge_index)
    data = VirtualNode()(data)
    data.y_graph = torch.randn(11)

    batch = Batch.from_data_list([data])
    model = GPSE()

    with torch.no_grad():
        out = model(batch)
        assert out[0].size() == out[1].size()
        assert out[0].size() == (7, 62)

    if is_full_test():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        min_loss = float('inf')
        for _ in range(100):
            optimizer.zero_grad()
            pred, true = model(batch)
            batch_idx = process_batch_idx(batch.batch, true)
            loss, _ = gpse_loss(pred, true, batch_idx)
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2


def test_gpse_from_pretrained():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)
    data = VirtualNode()(data)

    model = GPSE()
    model.post_mp = IdentityHead()

    with torch.no_grad():
        out = gpse_process(model, data, 'NormalSE')
        assert out.size() == (7, 512)


@pytest.mark.parametrize('expand_x', [False, True])
def test_gpse_node_encoder(expand_x):
    x = torch.randn(6, 4)
    pestat_GPSE = torch.randn(6, 512)

    encoder = GPSENodeEncoder(
        dim_emb=128,
        dim_pe_in=512,
        dim_pe_out=64,
        dim_in=4,
        expand_x=expand_x,
    )
    out = encoder(x, pestat_GPSE)
    assert out.size() == (6, 128) if expand_x else (6, 64)
