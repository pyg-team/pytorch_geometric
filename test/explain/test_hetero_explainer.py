import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    HeteroExplanation,
)
from torch_geometric.nn import SAGEConv, to_hetero


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    return data


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

    def forward(self, x_dict, edge_index_dict, *args) -> torch.Tensor:
        return self.graph_sage(x_dict, edge_index_dict)['paper']


def test_get_prediction():
    data = hetero_data()
    model = HeteroSAGE(data.metadata())
    assert model.training

    explainer = Explainer(
        model,
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='phenomenon',
            node_mask_type='object',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )
    pred = explainer.get_prediction(data.x_dict, data.edge_index_dict)
    assert model.training

    assert pred.size() == (8, 1)


def test_forward():
    data = hetero_data()
    model = HeteroSAGE(data.metadata())

    explainer = Explainer(
        model,
        algorithm=DummyExplainer(),
        explainer_config=dict(
            explanation_type='model',
            node_mask_type='attributes',
        ),
        model_config=dict(
            mode='regression',
            task_level='graph',
        ),
    )

    explanation = explainer(data.x_dict, data.edge_index_dict, is_hetero=True)
    assert isinstance(explanation, HeteroExplanation)
    # TODO: HeteroExplanation.available_explanations not working, unsure why
    # assert 'node_feat_mask' in explanation.available_explanations
    for node_type, node_feat_mask in explanation.node_feat_mask.items():
        assert node_feat_mask.size() == data[node_type].x.size()


# TODO: Add thresholding tests
