import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye

class PANPool(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.filter_size = filter_size
        if panpool_filter_weight is None:
            self.panpool_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, M=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Path integral
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)

        # weighted degree
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree = scatter_add(edge_weight, edge_index[0], out=degree)

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        x_transform_norm = xtransform #/ xtransform.norm(p=2, dim=-1)
        degree_norm = degree #/ degree.norm(p=2, dim=-1)
        score = self.pan_pool_weight[0] * x_transform_norm + self.pan_pool_weight[1] * degree_norm

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')