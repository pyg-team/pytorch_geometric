import torch


class MetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as global-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the functions :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    Args:
        edge_model (func, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (func, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (func, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features. To allow for
            batch-wise graph processing, the callable takes a fifth argument
            :obj:`batch`, which determines the assignment of nodes to their
            specific graphs. (default: :obj:`None`)

    .. code-block:: python

        from torch.nn import Sequential as Seq, Linear as Lin, ReLU
        from torch_scatter import scatter_mean
        from torch_geometric.nn import MetaLayer

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.node_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

                def edge_model(source, target, edge_attr, u):
                    # source, target: [E, F_x], where E is the number of edges.
                    # edge_attr: [E, F_e]
                    # u: [B, F_u], where B is the number of graphs.
                    out = torch.cat([source, target, edge_attr], dim=1)
                    return self.edge_mlp(out)

                def node_model(x, edge_index, edge_attr, u):
                    # x: [N, F_x], where N is the number of nodes.
                    # edge_index: [2, E] with max entry N - 1.
                    # edge_attr: [E, F_e]
                    # u: [B, F_u]
                    row, col = edge_index
                    out = torch.cat([x[col], edge_attr], dim=1)
                    out = self.node_mlp(out)
                    return scatter_mean(out, row, dim=0, dim_size=x.size(0))

                def global_model(x, edge_index, edge_attr, u, batch):
                    # x: [N, F_x], where N is the number of nodes.
                    # edge_index: [2, E] with max entry N - 1.
                    # edge_attr: [E, F_e]
                    # u: [B, F_u]
                    # batch: [N] with max entry B - 1.
                    out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                    return self.global_mlp(out)

                self.op = MetaLayer(edge_model, node_model, global_model)

            def forward(self, x, edge_index, edge_attr, u, batch):
                return self.op(x, edge_index, edge_attr, u, batch)
    """

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

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        """"""
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
