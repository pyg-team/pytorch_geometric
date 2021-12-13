import torch
from torch_geometric.nn.models import LightGCN


def test_lightgcn():
    emb_size = 5
    n_users = 3
    m_items = 100
    num_layers = 2
    model_config = {
        "n_users": n_users,
        "m_items": m_items,
        "embedding_size": emb_size,
        "num_layers": num_layers
    }
    lightGCN = LightGCN(model_config)
    edges = torch.tensor([[0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2],
                         [10, 20, 23, 34, 93,
                         10, 8, 94, 85, 78,
                         8, 60, 49, 38, 68]],
                         dtype=torch.long)
    all_users_items = lightGCN(
        lightGCN.embedding_user_item.weight.clone(),
        edges)

    assert lightGCN.embedding_user_item.weight.shape[0] ==\
        n_users + m_items
    assert lightGCN.embedding_user_item.weight.shape[1] ==\
        emb_size
    assert all_users_items.shape[0] == n_users + m_items
    assert all_users_items.shape[1] == emb_size
    assert str(lightGCN) ==\
        f'LightGCN({n_users}, {m_items}, " +\
            num_layers={num_layers})'
