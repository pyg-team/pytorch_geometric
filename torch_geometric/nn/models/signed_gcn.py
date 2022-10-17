from typing import Optional, Tuple

import scipy.sparse
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import coalesce

from torch_geometric.nn import SignedConv
from torch_geometric.utils import (
    negative_sampling,
    structured_negative_sampling,
)


class SignedGCN(torch.nn.Module):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
    Internally, this module uses the
    :class:`torch_geometric.nn.conv.SignedConv` operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of layers.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        lamb: float = 5,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb

        self.conv1 = SignedConv(in_channels, hidden_channels // 2,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def split_edges(
        self,
        edge_index: Tensor,
        test_ratio: float = 0.2,
    ) -> Tuple[Tensor, Tensor]:
        r"""Splits the edges :obj:`edge_index` into train and test edges.

        Args:
            edge_index (LongTensor): The edge indices.
            test_ratio (float, optional): The ratio of test edges.
                (default: :obj:`0.2`)
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def create_spectral_features(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, N, N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_channels, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def forward(
        self,
        x: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (Tensor): The input node features.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
    ) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float]:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0

        return auc, f1

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, num_layers={self.num_layers})')
