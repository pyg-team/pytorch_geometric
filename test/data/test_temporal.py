import torch

import torch_geometric
from torch_geometric.data import TemporalData


def _generate_temporal_data(num_events: int = 3, num_msg_features: int = 2):

    params = {
        'src': torch.arange(num_events),
        'dst': torch.arange(num_events, num_events * 2),
        't': torch.arange(0, num_events * 1000, step=1000),
        'msg': torch.randint(0, 2, (num_events, num_msg_features)),
        'y': torch.randint(0, 2, (num_events, )),
    }

    return TemporalData(**params).to(torch.device('cpu'))


def test_temporal_data():
    torch_geometric.set_debug(True)

    data = _generate_temporal_data()

    N = data.num_nodes
    assert N == 6

    num_events = data.num_events
    assert num_events == 3
    assert len(data) == 3

    assert data.src.tolist() == [0, 1, 2]
    assert data['src'].tolist() == [0, 1, 2]

    assert sorted(data.keys) == ['dst', 'msg', 'src', 't', 'y']

    D = data.to_dict()
    assert len(D) == 5
    for key in ['dst', 'msg', 'src', 't', 'y']:
        assert key in D

    D = data.to_namedtuple()
    assert len(D) == 5
    assert D.src is not None
    assert D.dst is not None
    assert D.t is not None
    assert D.msg is not None
    assert D.y is not None

    assert data.__cat_dim__('src', data.src) == 0
    assert data.__inc__('src', data.src) == 6

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.src.data_ptr() != data.src.data_ptr()
    assert clone.src.tolist() == data.src.tolist()
    assert clone.dst.data_ptr() != data.dst.data_ptr()
    assert clone.dst.tolist() == data.dst.tolist()

    assert str(
        data) == 'TemporalData(src=[3], dst=[3], t=[3], msg=[3, 2], y=[3])'

    key = value = 'test_value'
    data[key] = value
    assert data[key] == value
    del data[value]
    del data[value]  # Deleting unset attributes should work as well.

    assert data.get(key) is None

    torch_geometric.set_debug(False)


def test_train_val_test_split():
    torch_geometric.set_debug(True)

    num_events = 100
    data = _generate_temporal_data(num_events)

    train, val, test = data.train_val_test_split(val_ratio=0.2,
                                                 test_ratio=0.15)
    assert len(train) == num_events * 0.65
    assert len(val) == num_events * 0.2
    assert len(test) == num_events * 0.15

    assert max(train.t) < min(val.t)
    assert max(val.t) < min(test.t)

    torch_geometric.set_debug(False)


def test_temporal_get_item():
    torch_geometric.set_debug(True)

    num_events = 10
    data = _generate_temporal_data(num_events)

    item = data[0]
    assert type(item) == TemporalData
    assert len(item) == 1
    assert item.src == data.src[0]
    assert item.dst == data.dst[0]
    assert torch.all(item.msg == data.msg[0])
    assert item.t == data.t[0]

    item = data[0:5]
    assert type(item) == TemporalData
    assert len(item) == 5
    assert torch.all(item.src == data.src[0:5])
    assert torch.all(item.dst == data.dst[0:5])
    assert torch.all(item.msg == data.msg[0:5])
    assert torch.all(item.t == data.t[0:5])

    item = data[(0, 4, 8)]
    assert type(item) == TemporalData
    assert len(item) == 3
    assert torch.all(item.src == data.src[0::4])
    assert torch.all(item.dst == data.dst[0::4])
    assert torch.all(item.msg == data.msg[0::4])
    assert torch.all(item.t == data.t[0::4])

    item = data[(True, False, True, False, True, False, True, False, True,
                 False)]
    assert type(item) == TemporalData
    assert len(item) == 5
    assert torch.all(item.src == data.src[0::2])
    assert torch.all(item.dst == data.dst[0::2])
    assert torch.all(item.msg == data.msg[0::2])
    assert torch.all(item.t == data.t[0::2])

    torch_geometric.set_debug(False)
