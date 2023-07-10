import torch

from torch_geometric.data import Batch, TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


def test_tgn():
    memory_dim = 16
    time_dim = 16

    src = torch.tensor([0, 1, 0, 2, 0, 3, 1, 4, 2, 3])
    dst = torch.tensor([1, 2, 1, 1, 3, 2, 4, 3, 3, 4])
    t = torch.arange(10)
    msg = torch.randn(10, 16)
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    loader = TemporalDataLoader(data, batch_size=5, neighbor_loader_size=3)

    assert loader.neighbor_loader.cur_e_id == 0
    assert loader.neighbor_loader.e_id.size() == (data.num_nodes, 3)

    memory = TGNMemory(
        num_nodes=data.num_nodes,
        raw_msg_dim=msg.size(-1),
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    )
    assert memory.memory.size() == (data.num_nodes, memory_dim)
    assert memory.last_update.size() == (data.num_nodes, )

    # Test TGNMemory training:
    for i, batch in enumerate(loader):
        z, last_update = memory(batch.n_id)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        if i == 0:
            assert batch.n_id.size(0) == 4
            assert batch.edge_index.numel() == 0
            assert batch.e_id.numel() == 0
            assert z.size() == (batch.n_id.size(0), memory_dim)
            assert torch.sum(last_update) == 0
        else:
            assert batch.n_id.size(0) == 5
            assert batch.edge_index.numel() == 12
            assert batch.e_id.numel() == 6
            assert z.size() == (batch.n_id.size(0), memory_dim)
            assert torch.equal(last_update, torch.tensor([4, 3, 3, 4, 0]))

    # Test TGNMemory inference:
    memory.eval()
    all_n_id = torch.arange(data.num_nodes)
    z, last_update = memory(all_n_id)
    assert z.size() == (data.num_nodes, memory_dim)
    assert torch.equal(last_update, torch.tensor([4, 6, 8, 9, 9]))

    batch = Batch(src=torch.tensor([3, 4]), dst=torch.tensor([4, 3]),
                  t=torch.tensor([10, 10]), msg=torch.randn(2, 16))
    memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
    post_z, post_last_update = memory(all_n_id)
    assert torch.allclose(z[0:3], post_z[0:3])
    assert torch.equal(post_last_update, torch.tensor([4, 6, 8, 10, 10]))

    memory.reset_state()
    assert memory.memory.sum() == 0
    assert memory.last_update.sum() == 0
