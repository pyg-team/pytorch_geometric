import torch
import torch_geometric


class PointPairFeatures(object):
    r"""Computes the rotation-invariant Point Pair Features

    .. math::
        \left( \| \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
        \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
        \mathbf{n}_j) \right)

    of linked nodes in its edge attributes, where :math:`\mathbf{d}_{j,i}`
    denotes the difference vector between, and :math:`\mathbf{n}_i` and
    :math:`\mathbf{n}_j` denote the surface normals of node :math:`i` and
    :math:`j` respectively.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        ppf_func = torch_geometric.nn.conv.ppf_conv.point_pair_features

        assert data.edge_index is not None
        assert data.pos is not None and data.norm is not None
        assert data.pos.size(-1) == 3
        assert data.pos.size() == data.norm.size()

        row, col = data.edge_index
        pos, norm, pseudo = data.pos, data.norm, data.edge_attr

        ppf = ppf_func(pos[row], pos[col], norm[row], norm[col])

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, ppf.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = ppf

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
