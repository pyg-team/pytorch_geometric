import pytest
import torch

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader


@pytest.mark.parametrize('batch_size,drop_last', [(4, True), (2, False)])
def test_temporal_dataloader(batch_size, drop_last):
    src = dst = t = torch.arange(10)
    msg = torch.randn(10, 16)

    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    loader = TemporalDataLoader(
        data,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    assert len(loader) == 10 // batch_size

    for i, batch in enumerate(loader):
        assert len(batch) == batch_size
        arange = range(len(batch) * i, len(batch) * i + len(batch))
        assert batch.src.tolist() == data.src[arange].tolist()
        assert batch.dst.tolist() == data.dst[arange].tolist()
        assert batch.t.tolist() == data.t[arange].tolist()
        assert batch.msg.tolist() == data.msg[arange].tolist()
