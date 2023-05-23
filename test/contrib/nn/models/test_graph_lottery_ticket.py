from copy import deepcopy

import pytest
import torch
from torch.nn import Module
from torch.optim import Adadelta

from torch_geometric.contrib.nn import GLTMask, GLTModel, GLTSearch
from torch_geometric.data import Data
from torch_geometric.nn import GCN
from torch_geometric.utils import negative_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 3],
                           [1, 3], [3, 1], [0, 5], [5, 0]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [-1], [0], [1]], dtype=torch.float)
y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float).unsqueeze(1)

train_mask = torch.tensor([True, True, False, False, False, False])
val_mask = torch.tensor([False, False, True, True, False, False])
test_mask = torch.tensor([False, False, False, False, True, True])


def generate_edge_data(dataset):
    """sample  negative edges and labels"""
    negative_edges = negative_sampling(
        dataset.edge_index, num_neg_samples=2 * dataset.edge_index.shape[1])
    edge_labels = [0] * negative_edges.shape[-1] + [
        1
    ] * dataset.edge_index.shape[-1]
    dataset.edges = torch.cat([dataset.edge_index, negative_edges], dim=-1)
    dataset.edge_labels = torch.tensor(edge_labels).to(device)


test_data_node_classification = Data(x=x, y=y,
                                     edge_index=edge_index.t().contiguous(),
                                     train_mask=train_mask, val_mask=val_mask,
                                     test_mask=test_mask).to(device)
test_data_link_prediction = Data(x=x, edge_index=edge_index.t().contiguous(),
                                 train_mask=train_mask, val_mask=val_mask,
                                 test_mask=test_mask).to(device)
generate_edge_data(test_data_link_prediction)


class LinkPredictor(Module):
    """helper model to get dot product interaction on edges"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_weight, edges):
        x = self.model(x, edge_index, edge_weight=edge_weight)
        edge_feat_i = x[edges[0]]
        edge_feat_j = x[edges[1]]
        return (edge_feat_i * edge_feat_j).sum(dim=-1)


def build_test_model(architecture, link_prediction):
    input_dim = 1
    hidden_dim = 5
    if link_prediction:
        model = architecture(in_channels=input_dim, hidden_channels=hidden_dim,
                             out_channels=hidden_dim, num_layers=2)
        model = LinkPredictor(model)
    else:
        model = architecture(in_channels=input_dim, hidden_channels=hidden_dim,
                             out_channels=1, num_layers=2)
    model.to(device)
    return model


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_make_mask(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    adjacency_shape = graph.edge_index.shape[1]

    model = build_test_model(architecture, link_prediction)
    mask = GLTMask(model, graph, torch.device(device)).to_dict()

    for name, param in model.named_parameters():
        param_mask = mask[name + GLTModel.MASK]
        assert param.shape == param_mask.shape

    assert mask[GLTModel.EDGE_MASK + GLTModel.MASK].shape[0] == \
           adjacency_shape


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_apply_mask(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    model = build_test_model(architecture, link_prediction)
    pruned_model = GLTModel(model, graph, ignore_keys=set())
    mask = GLTMask(model, graph, torch.device(device)).to_dict()

    original_params = deepcopy(dict(model.named_parameters()))
    pruned_model.apply_mask(mask)
    new_params = dict(model.named_parameters())

    for name, param in original_params.items():
        assert param.shape == new_params[name + GLTModel.MASK].shape
        assert param.shape == new_params[name + GLTModel.ORIG].shape


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_apply_mask_ignore_keys(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    model = build_test_model(architecture, link_prediction)
    ignore_keys = {'bias'}
    pruned_model = GLTModel(model, graph, ignore_keys=ignore_keys)
    mask = GLTMask(model, graph, torch.device(device)).to_dict()

    original_params = deepcopy(dict(model.named_parameters()))
    pruned_model.apply_mask(mask)
    new_params = dict(model.named_parameters())

    for name, param in original_params.items():
        key = name.rpartition('.')[-1]
        if key in ignore_keys:
            assert name in new_params
            assert name + GLTModel.MASK not in new_params
            assert name + GLTModel.ORIG not in new_params


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_forward(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
        input_len = graph.edges.shape[1]
    else:
        graph = test_data_node_classification
        input_len = graph.x.shape[0]

    model = build_test_model(architecture, link_prediction)
    pruned_model = GLTModel(model, graph, ignore_keys=set())
    mask = GLTMask(model, graph, torch.device(device)).to_dict()

    pruned_model.apply_mask(mask)
    output = pruned_model()
    assert output.shape[0] == input_len


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_rewind(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    model = build_test_model(architecture, link_prediction)
    initial_params = {
        "module." + k + GLTModel.ORIG if k.rpartition(".")[-1] else "module." +
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }
    pruned_model = GLTModel(model, graph, ignore_keys=set())
    mask = GLTMask(model, graph, torch.device(device))

    pruned_model.apply_mask(mask.to_dict())
    pruned_model.rewind({**mask.to_dict(weight_prefix=True), **initial_params})

    for param in ({
            **mask.to_dict(weight_prefix=True),
            **initial_params
    }).values():
        assert not param.requires_grad
        assert param.grad is None


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
@pytest.mark.parametrize('ugs', [True, False])
def test_search_train(architecture, link_prediction, ugs):
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    model = build_test_model(architecture, link_prediction)
    mask_names = [name + GLTModel.MASK for name, _ in model.named_parameters()]

    search = GLTSearch(
        module=model, device=torch.device(device), graph=graph, lr=0.001, reg_graph=0.01, reg_model=0.01,
        optim_args={
            'weight_decay': 8e-5,
            'rho': 0.8
        },
        task='link_prediction' if link_prediction else 'node_classification',
        optimizer=Adadelta, max_train_epochs=2)

    ticket = GLTModel(search.module, search.graph,
                      ignore_keys=search.ignore_keys)
    ticket.apply_mask(search.mask.to_dict())
    final_test_score, best_masks = search.train(ticket, ugs)

    if not ugs:
        assert len(best_masks) == 0
    else:
        for name in mask_names:
            assert name in best_masks.keys()


@pytest.mark.parametrize('architecture', [GCN])
@pytest.mark.parametrize('link_prediction', [False, True])
def test_search_prune(architecture, link_prediction):
    if link_prediction:
        graph = test_data_link_prediction
    else:
        graph = test_data_node_classification

    model = build_test_model(architecture, link_prediction)
    params = model.state_dict()

    search = GLTSearch(
        module=model, device=torch.device(device), graph=graph, lr=0.001, reg_graph=0.01, reg_model=0.01,
        task='link_prediction' if link_prediction else 'node_classification',
        max_train_epochs=2, optim_args={})

    init_params, mask_dict, _ = search.prune()

    for name, value in params.items():
        mask_name = name + GLTModel.MASK
        assert mask_name in mask_dict.keys()
        assert value.shape == mask_dict[mask_name].shape
        assert 'module.' + name + GLTModel.ORIG in init_params.keys()
