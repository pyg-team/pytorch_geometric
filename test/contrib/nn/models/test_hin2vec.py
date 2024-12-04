import pytest
import torch

from torch_geometric.contrib.nn import HIN2Vec
from torch_geometric.testing import has_package, withDevice
from torch_geometric.typing import WITH_TORCH_CLUSTER


@withDevice
@pytest.mark.parametrize('reg', ['sigmoid', 'step'])
def test_hin2vec(device, reg):
    edge_index_dict = {
        ('author', 'writes', 'paper'):
        torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 1, 0]], device=device),
        ('paper', 'written_by', 'author'):
        torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 2, 3]], device=device)
    }
    kwargs = dict(embedding_dim=16, metapath_length=2, walk_length=3,
                  walks_per_node=4, reg=reg)

    if not WITH_TORCH_CLUSTER:
        with pytest.raises(ImportError,
                           match=("requires the 'torch-cluster'")):
            model = HIN2Vec(edge_index_dict, **kwargs)
        return

    model = HIN2Vec(edge_index_dict, **kwargs).to(device)
    assert str(model) == 'HIN2Vec(6, 16)'

    node_emb, path_emb = model("author")
    assert node_emb.size() == (4, 16)
    assert 2 <= path_emb.size(0) <= 4
    assert path_emb.size(1) == 16

    node_emb, _ = model("paper")
    assert node_emb.size() == (2, 16)

    node_emb, _ = model("author", torch.arange(2, device=device))
    assert node_emb.size() == (2, 16)

    loader = model.loader(batch_size=4, shuffle=True)
    pos_sample, neg_sample = next(iter(loader))

    loss = model.loss(pos_sample.to(device), neg_sample.to(device))
    assert loss.item() >= 0

    if has_package('sklearn'):
        z, _ = model("author")
        acc = model.test(z[:3], torch.tensor([2, 1, 2]), z[3:],
                         torch.tensor([1]))
        assert 0 <= acc <= 1
