import copy

import torch

from torch_geometric.data import TemporalData


def get_temporal_data(num_events, msg_channels):
    return TemporalData(
        src=torch.arange(num_events),
        dst=torch.arange(num_events, num_events * 2),
        t=torch.arange(0, num_events * 1000, step=1000),
        msg=torch.randn(num_events, msg_channels),
        y=torch.randint(0, 2, (num_events, )),
    )


def test_temporal_data():
    data = get_temporal_data(num_events=3, msg_channels=16)
    assert str(data) == ("TemporalData(src=[3], dst=[3], t=[3], "
                         "msg=[3, 16], y=[3])")

    assert data.num_nodes == 6
    assert data.num_events == data.num_edges == len(data) == 3

    assert data.src.tolist() == [0, 1, 2]
    assert data['src'].tolist() == [0, 1, 2]

    assert data.edge_index.tolist() == [[0, 1, 2], [3, 4, 5]]
    data.edge_index = 'edge_index'
    assert data.edge_index == 'edge_index'
    del data.edge_index
    assert data.edge_index.tolist() == [[0, 1, 2], [3, 4, 5]]

    assert sorted(data.keys) == ['dst', 'msg', 'src', 't', 'y']
    assert sorted(data.to_dict().keys()) == sorted(data.keys)

    data_tuple = data.to_namedtuple()
    assert len(data_tuple) == 5
    assert data_tuple.src is not None
    assert data_tuple.dst is not None
    assert data_tuple.t is not None
    assert data_tuple.msg is not None
    assert data_tuple.y is not None

    assert data.__cat_dim__('src', data.src) == 0
    assert data.__inc__('src', data.src) == 6

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.src.data_ptr() != data.src.data_ptr()
    assert clone.src.tolist() == data.src.tolist()
    assert clone.dst.data_ptr() != data.dst.data_ptr()
    assert clone.dst.tolist() == data.dst.tolist()

    deepcopy = copy.deepcopy(data)
    assert deepcopy != data
    assert len(deepcopy) == len(data)
    assert deepcopy.src.data_ptr() != data.src.data_ptr()
    assert deepcopy.src.tolist() == data.src.tolist()
    assert deepcopy.dst.data_ptr() != data.dst.data_ptr()
    assert deepcopy.dst.tolist() == data.dst.tolist()

    key = value = 'test_value'
    data[key] = value
    assert data[key] == value
    assert data.test_value == value
    del data[key]
    del data[key]  # Deleting unset attributes should work as well.

    assert data.get(key, 10) == 10

    assert len([event for event in data]) == 3

    assert len([attr for attr in data()]) == 5

    assert data.size() == (2, 5)

    del data.src
    assert 'src' not in data


def test_train_val_test_split():
    data = get_temporal_data(num_events=100, msg_channels=16)

    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.2, test_ratio=0.15)

    assert len(train_data) == 65
    assert len(val_data) == 20
    assert len(test_data) == 15

    assert train_data.t.max() < val_data.t.min()
    assert val_data.t.max() < test_data.t.min()


def test_temporal_indexing():
    data = get_temporal_data(num_events=10, msg_channels=16)

    elem = data[0]
    assert isinstance(elem, TemporalData)
    assert len(elem) == 1
    assert elem.src.tolist() == data.src[0:1].tolist()
    assert elem.dst.tolist() == data.dst[0:1].tolist()
    assert elem.t.tolist() == data.t[0:1].tolist()
    assert elem.msg.tolist() == data.msg[0:1].tolist()
    assert elem.y.tolist() == data.y[0:1].tolist()

    subset = data[0:5]
    assert isinstance(subset, TemporalData)
    assert len(subset) == 5
    assert subset.src.tolist() == data.src[0:5].tolist()
    assert subset.dst.tolist() == data.dst[0:5].tolist()
    assert subset.t.tolist() == data.t[0:5].tolist()
    assert subset.msg.tolist() == data.msg[0:5].tolist()
    assert subset.y.tolist() == data.y[0:5].tolist()

    index = [0, 4, 8]
    subset = data[torch.tensor(index)]
    assert isinstance(subset, TemporalData)
    assert len(subset) == 3
    assert subset.src.tolist() == data.src[0::4].tolist()
    assert subset.dst.tolist() == data.dst[0::4].tolist()
    assert subset.t.tolist() == data.t[0::4].tolist()
    assert subset.msg.tolist() == data.msg[0::4].tolist()
    assert subset.y.tolist() == data.y[0::4].tolist()

    mask = [True, False, True, False, True, False, True, False, True, False]
    subset = data[torch.tensor(mask)]
    assert isinstance(subset, TemporalData)
    assert len(subset) == 5
    assert subset.src.tolist() == data.src[0::2].tolist()
    assert subset.dst.tolist() == data.dst[0::2].tolist()
    assert subset.t.tolist() == data.t[0::2].tolist()
    assert subset.msg.tolist() == data.msg[0::2].tolist()
    assert subset.y.tolist() == data.y[0::2].tolist()
