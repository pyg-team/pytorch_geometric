from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCNNorm(object):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper.

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __call__(self, data):
        assert data.edge_index is not None or data.adj_t is not None

        if data.edge_index is not None:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index, edge_weight, data.num_nodes)
        else:
            data.adj_t = gcn_norm(data.adj_t)

        return data
