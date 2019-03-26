import math
import random

import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from ..inits import reset


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

    def decode(self, z):
        r"""Decodes the latent variables :obj:`z` into a probabilistic
        dense adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)

    def decode_indices(self, z, edge_index):
        r"""Decodes the latent variables :obj:`z` into edge-probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (LongTensor): The edge indices to predict.
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

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

    def reconstruction_loss(self, z, pos_edge_index, neg_adj_mask):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and a negative
        adjacency matrix mask :obj:`neg_adj_mask`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_adj_mask (ByteTensor): A symmetric mask with shape
                :obj:`[N, N]` denoting the negative edges to train against.
        """

        pos_loss = -torch.log(self.decode_indices(z, pos_edge_index)).mean()

        neg_loss = -torch.log(
            (1 - self.decode(z)[neg_adj_mask]).clamp(min=1e-8)).mean()

        return pos_loss + neg_loss

    def loss(self, z, pos_edge_index, neg_adj_mask):
        return self.reconstruction_loss(z, pos_edge_index, neg_adj_mask)

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
    def __init__(self, encoder):
        super(VGAE, self).__init__(encoder)

    def sample(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def loss(self, mu, logvar, pos_edge_index, neg_adj_mask):
        z = self.sample(mu, logvar)
        recon_loss = self.reconstruction_loss(z, pos_edge_index, neg_adj_mask)
        kl_loss = self.kl_loss(mu, logvar)
        return recon_loss + kl_loss


class ARGA(GAE):
    def __init__(self, encoder, discriminator):
        super(ARGA, self).__init__(encoder)
        self.discriminator = discriminator

    def reset_parameters(self):
        super(ARGA, self).reset_parameters(self)
        reset(self.discriminator)

    def discriminate(self, z):
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z))
        return real, fake

    def discriminator_loss(self, real, fake):
        real_loss = -torch.log(real).mean()
        fake_loss = -torch.log((1 - fake).clamp(min=1e-8)).mean()
        return real_loss + fake_loss

    def loss(self, z, pos_edge_index, neg_adj_mask):
        recon_loss = self.reconstruction_loss(z, pos_edge_index, neg_adj_mask)
        d_loss = self.discriminator_loss(*self.discriminate(z))
        return recon_loss + d_loss


class ARGVA(ARGA):
    def __init__(self, encoder, discriminator):
        super(ARGVA, self).__init__(encoder, discriminator)
        self.VGAE = VGAE(encoder)

    def sample(self, mu, logvar):
        return self.VGAE.sample(mu, logvar)

    def kl_loss(self, mu, logvar):
        return self.VGAE.kl_loss(mu, logvar)

    def loss(self, mu, logvar, pos_edge_index, neg_adj_mask):
        z = self.sample(mu, logvar)
        recon_loss = self.reconstruction_loss(z, pos_edge_index, neg_adj_mask)
        kl_loss = self.kl_loss(mu, logvar)
        d_loss = self.discriminator_loss(*self.discriminate(z))
        return recon_loss + kl_loss + d_loss
