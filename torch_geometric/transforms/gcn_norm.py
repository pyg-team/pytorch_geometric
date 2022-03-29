import torch_geometric
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('gcn_norm')
class GCNNorm(BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def __call__(self, data):
        gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
        assert 'edge_index' in data or 'adj_t' in data

        if 'edge_index' in data:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index, edge_weight, data.num_nodes,
                add_self_loops=self.add_self_loops)
        else:
            data.adj_t = gcn_norm(data.adj_t,
                                  add_self_loops=self.add_self_loops)

        return data
