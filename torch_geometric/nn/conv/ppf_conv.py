import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, scatter_
from ..inits import reset

class PPFConv(torch.nn.Module):
    r"""The PPF-Net operator from `"PPFNet: Global Context Aware Local Features for Robust 3D Point Matching"
    <https://arxiv.org/pdf/1802.02669.pdf>`

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \Vert l \Vert \angle(n1, d) \Vert \angle(n2, d) \Vert \angle(n1,n2)  )\right)

    where :math:`\gamma_{\mathbf{\Theta}}` and
    :math:`h_{\mathbf{\Theta}}` denote neural
    networks, *.i.e.* MLPs, :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    defines the position of each point, :math:`d' the vector defining the distance between
    two points, :math:`l' the normalized length of :math:`d', 
    :math:`n1' and :math:`n2' the normal vectors of the points.
    The angle calculation is defined by :math:`\angle(v_1, v_2) = atan2(\lVert v_1 \times v_2 \rVert, v_1 \cdot v_2)'.

    Args:
        local_nn (nn.Sequential, optional): Neural network
            :math:`h_{\mathbf{\Theta}}`. (default: :obj:`None`)
        global_nn (nn.Sequential, optional): Neural network
            :math:`\gamma_{\mathbf{\Theta}}`. (default: :obj:`None`)
    """

    def __init__(self, local_nn=None, global_nn=None):
        super(PPFConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()

    def get_angle(self, v1, v2):
        return torch.atan2(torch.cross(v1, v2, dim=1).norm(dim=1), (v1 * v2).sum(dim=1))

    def getPointFeatures(self, pos, norm, edge_index):
        row, col = edge_index
        n1 = norm[row]
        n2 = norm[col]
        d = pos[col] - pos[row]

        # normalize distance vector
        dist = d.pow(2).sum(1)        
        dist = torch.div(dist, dist.sum(0)/d.size(0))

        pf = torch.stack([  dist,
                            self.get_angle(n1, d), 
                            self.get_angle(n2, d), 
                            self.get_angle(n1, n2)])   
        return pf
    
    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, edge_index, norm, batch):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=pos.size(0))
        row, col = edge_index

        ppf = self.getPointFeatures(pos, norm, edge_index)
        out = ppf.t()
                        
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            out = torch.cat([x[col], out], dim=1)        

        if self.local_nn is not None:
            out = self.local_nn(out)

        out = scatter_('max', out, row, dim_size=pos.size(0))

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)
