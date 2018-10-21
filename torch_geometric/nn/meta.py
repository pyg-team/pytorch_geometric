import torch


class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, global_x=None,
                batch=None):
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, global_x)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, global_x)

        if self.global_model is not None:
            global_x = self.global_model(x, edge_index, edge_attr, global_x,
                                         batch)

        return x, edge_attr, global_x

    def __repr__(self):
        return '{}(edge_model={}, node_model={}, global_model={})'.format(
            self.__class__.__name__, self.edge_model, self.node_model,
            self.global_model)
