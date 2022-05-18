import torch

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader


def test_temporal_dataloader():
    src = dst = t = torch.arange(10)
    msg = torch.randn(10, 16)

    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    loader = TemporalDataLoader(data, batch_size=2)
    assert len(loader) == 5

    for i, batch in enumerate(loader):
        assert len(batch) == 2
        arange = range(len(batch) * i, len(batch) * i + len(batch))
        assert batch.src.tolist() == data.src[arange].tolist()
        assert batch.dst.tolist() == data.dst[arange].tolist()
        assert batch.t.tolist() == data.t[arange].tolist()
        assert batch.msg.tolist() == data.msg[arange].tolist()
