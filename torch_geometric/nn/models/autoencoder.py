import math
import random

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from ..inits import reset

EPS = 1e-15


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on a user-defined encoder model and a simple inner product
    decoder :math:`\sigma(\mathbf{Z}\mathbf{Z}^{\top})` where
    :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent space
    produced by the encoder.

    Args:
        encoder (Module): The encoder module.
    """

    def __init__(self, encoder):
        super(GAE, self).__init__()
        self.encoder = encoder

    def reset_parameters(self):
        reset(self.encoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes latent variables for each node."""
        return self.encoder(*args, **kwargs)

    def decode(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic
        dense adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

    def decode_indices(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge-probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (LongTensor): The edge indices to predict.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = math.floor(val_ratio * row.size(0))
        n_t = math.floor(test_ratio * row.size(0))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def negative_sampling(self, pos_edge_index, num_nodes):
        idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
        idx = idx.to(torch.device('cpu'))

        rng = range(num_nodes**2)
        perm = torch.tensor(random.sample(rng, idx.size(0)))
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero().view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
            perm[rest] = tmp
            rest = mask.nonzero().view(-1)

        row, col = perm / num_nodes, perm % num_nodes
        return torch.stack([row, col], dim=0).to(pos_edge_index.device)

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(self.decode_indices(z, pos_edge_index) +
                              EPS).mean()

        neg_edge_index = self.negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decode_indices(z, neg_edge_index) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decode_indices(z, pos_edge_index)
        neg_pred = self.decode_indices(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log \sigma^2`.
    """

    def __init__(self, encoder):
        super(VGAE, self).__init__(encoder)

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        r""""""
        self.mu, self.logvar = self.encoder(*args, **kwargs)
        z = self.reparametrize(self.mu, self.logvar)
        return z

    def kl_loss(self, mu=None, logvar=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.mu if mu is None else mu
        logvar = self.logvar if logvar is None else logvar
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module.
        discriminator (Module): The discriminator module.
    """

    def __init__(self, encoder, discriminator):
        super(ARGA, self).__init__(encoder)
        self.discriminator = discriminator

    def reset_parameters(self):
        super(ARGA, self).reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log \sigma^2`.
        discriminator (Module): The discriminator module.
    """

    def __init__(self, encoder, discriminator):
        super(ARGVA, self).__init__(encoder, discriminator)
        self.VGAE = VGAE(encoder)

    @property
    def mu(self):
        return self.VGAE.mu

    @property
    def logvar(self):
        return self.VGAE.logvar

    def reparametrize(self, mu, logvar):
        return self.VGAE.reparametrize(mu, logvar)

    def encode(self, *args, **kwargs):
        r""""""
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logvar=None):
        return self.VGAE.kl_loss(mu, logvar)
