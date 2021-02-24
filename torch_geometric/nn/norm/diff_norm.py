import torch
from torch import nn, Tensor, cdist
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d


class DiffNorm(nn.Module):
    r"""Applies a differentiable group norm. Based on
    `"Towards Deeper Graph Neural Networks with Differentiable
    Group Normalization"
    <https://arxiv.org/abs/2006.06972>`_paper.
    Args:
        in_channels(int): number of features.
        groups(int): number of groups nodes will be clustered into.
                     For classification tasks this could be number of classes.
        lambda(float): balancing factor used while adding normalized embedding
                       to current embedding.
        momentum(float): momentum for batch norm.
    """
    def __init__(self, in_channels=2, groups=3, lamda=0.01, momentum=0.3):
        super().__init__()

        self.G = groups
        self.nd = in_channels
        self.lamda = lamda
        self.U = Linear(self.nd, self.G, bias=False)
        self.bn = BatchNorm1d(self.G * self.nd, momentum=momentum)

    def groupDistanceRatio(self, h: Tensor, c: Tensor,
                           eps: float = 1e-9) -> float:
        r"""Measures oversmoothing in h. Assumes nodes of the same class belong
        to a cluster. Then computes ratio of average inter cluster distance to
        intra cluster distance.
        Args:
            h(N*F): node embeddings.
            c(N*1): node class.
            eps: Small value to avoid division by zero error.
        """
        device = h.device
        if c.dim() == 1:
            c = c.unsqueeze(1)
        if h.dim() == 1:
            h = h.unsqueeze(1)

        bin_count = torch.bincount(
            torch.cat([
                c.squeeze(1),
                torch.arange(0,
                             int(max(c)) + 1, device=device, dtype=int)
            ])) - 1
        # counts[i]=count(class(node i)) .shape=N
        counts = torch.index_select(bin_count, 0, c.squeeze(1))

        # denom[i][j]=1/(count(class(node i))*count(class(node i)))
        denom = torch.pow(
            counts.repeat(len(c), 1).float() * counts.repeat(len(c), 1).T,
            -1.0)

        # pair_dist[i][j]=||hi-hj||
        pair_dist = cdist(h, h, p=2)

        pair_dist = pair_dist * denom
        del denom, counts, bin_count

        # same_class[i][j]=True if node i and node j belong to same class.
        same_class = c.repeat(1, len(h)) == c.repeat(1, len(h)).T

        inter_dist = torch.sum(
            torch.where(~same_class, pair_dist,
                        torch.zeros(pair_dist.shape, device=device)))
        intra_dist = torch.sum(
            torch.where(same_class, pair_dist,
                        torch.zeros(pair_dist.shape, device=device)))
        del same_class

        C = len(torch.unique(c))
        return float(
            (C / (eps + C - 1.0)**2) * (inter_dist / (eps + intra_dist)))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x(N*nd): Node embedding.
        """
        N = len(x)
        S = F.softmax(self.U(x), dim=1)  # dim=N*G
        S_broad = S.T.reshape(self.G, N, 1).repeat(1, 1, self.nd)  # dim=G*N*nd

        xg = (S_broad * x).permute(1, 0, 2).reshape(N, self.G * self.nd)
        xg = self.bn(xg)

        # summing over groups
        sum_xg = xg.view(-1, self.G, self.nd).sum(1)

        return x + self.lamda * sum_xg
