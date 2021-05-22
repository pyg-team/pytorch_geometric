import torch
from torch_geometric.data import TemporalData, TemporalDataset, \
    TemporalDataLoader


def test_temporal_dataloader():
    src = torch.tensor([1, 2, 0])
    dst = torch.tensor([3, 1, 2])
    t = torch.Tensor([0., 5., 29.])
    msg = torch.Tensor([[564., 8997., 22., 5.], [8965., 53., 5., 892.],
                        [879., 23., 564., 88.]])
    y = torch.tensor([0, 1, 0])
    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

    assert data.__repr__() == (
        'TemporalData(dst=[3], msg=[3, 4], src=[3], t=[3], y=[3])')

    dataset = TemporalDataset(data)

    loader = TemporalDataLoader(dataset, batch_size=2, shuffle=False)

    for batch in loader:
        assert len(batch) == len(batch.keys)
        assert batch.src.tolist() == [1, 2]
        assert batch.dst.tolist() == [3, 1]
        assert batch.t.tolist() == [0., 5.]
        assert batch.msg.tolist() == [[564., 8997., 22., 5.],
                                      [8965., 53., 5., 892.]]
        assert batch.y.tolist() == [0, 1]
        break
