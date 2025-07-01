import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.models.glem import GLEM
from torch_geometric.testing import withPackage

@withPackage('transformers')
def test_glem_loss_handles_nan():
    gnn = GraphSAGE(8, 16, num_layers=3, out_channels=47)
    model = GLEM(gnn_to_use=gnn)

    logits = torch.tensor([[float('nan'), 0.0], [1.0, -1.0]],
                          requires_grad=True)
    labels = torch.tensor([0, 1])
    is_gold = torch.tensor([True, False])
    pseudo_labels = torch.tensor([1, 0])
    loss_func = torch.nn.CrossEntropyLoss()

    loss = model.loss(
        logits=logits,
        labels=labels,
        loss_func=loss_func,
        is_gold=is_gold,
        pseudo_labels=pseudo_labels,
        pl_weight=0.5,
        is_augmented=True,
    )

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.requires_grad
