import torch

import torch_geometric
from torch_geometric.data import TemporalData


def test_temporal_data():
    torch_geometric.set_debug(True)

    params = {
        'src': torch.tensor([1, 2, 3]),
        'dst': torch.tensor([4, 5, 6]),
        't': torch.tensor([1000, 2000, 3000]),
        'msg': torch.tensor([[0, 1], [1, 1], [0, 0]]),
        'y': torch.tensor([1, 1, 0])
    }

    data = TemporalData(**params).to(torch.device('cpu'))

    N = data.num_nodes
    assert N == 6

    num_events = data.num_events
    assert num_events == 3

    assert data.src.tolist() == params['src'].tolist()
    assert data['src'].tolist() == params['src'].tolist()

    assert sorted(data.keys) == ['dst', 'msg', 'src', 't', 'y']
    assert len(data) == 5

    D = data.to_dict()
    assert len(D) == 5
    for key in params.keys():
        assert key in D

    D = data.to_namedtuple()
    assert len(D) == 5
    assert D.src is not None
    assert D.dst is not None
    assert D.t is not None
    assert D.msg is not None
    assert D.y is not None

    assert data.__cat_dim__('src', data.src) == 0
    assert data.__inc__('src', data.src) == 0

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
