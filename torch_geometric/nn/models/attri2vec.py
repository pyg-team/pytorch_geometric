from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import WITH_PYG_LIB
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


class Attri2Vec(torch.nn.Module):
    r"""The attri2vec model from the `"Attributed Network Embedding
    via Subspace Discovery" <https://arxiv.org/abs/1901.04095>`_ paper.

    `attri2vec` learns node embeddings by mapping a node's features into a
    structure-aware subspace using a linear or non-linear transformation.
    It optimizes the embedding space by predicting context nodes collected
    from random walks.

    Args:
        edge_index (torch.Tensor): The edge indices.
        x (torch.Tensor): The node features.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size considered for
            positive samples.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        mapping (str, optional): The transformation function applied to node
            features (:obj:`"linear"`, :obj:`"relu"`, :obj:`"kernel"`, or
            :obj:`"sigmoid"`). (default: :obj:`"linear"`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        edge_index: Tensor,
        x: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        mapping: str = 'linear',
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if not WITH_PYG_LIB:
            raise ImportError(f"'{self.__class__.__name__}' "
                              f"requires 'pyg-lib>=0.6.0'")
        self.random_walk_fn = torch.ops.pyg.random_walk
        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size
        assert mapping in ('linear', 'relu', 'sigmoid', 'kernel')

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.mapping = mapping
        self._x: Tensor
        self.register_buffer('_x', x)

        num_features = x.size(1)

        if mapping == 'kernel':
            assert embedding_dim % 2 == 0
            self.lin = Linear(num_features, embedding_dim // 2, bias=False)
        else:
            self.lin = Linear(num_features, embedding_dim, bias=False)

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    @property
    def x(self) -> Tensor:
        return self._x

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        x = self.x if batch is None else self.x[batch]
        return self._transform(x)

    def _transform(self, x: Tensor) -> Tensor:
        r"""Apply f(x): the attribute-to-embedding transformation."""
        if self.mapping == 'linear':
            return self.lin(x)
        elif self.mapping == 'relu':
            return F.relu(self.lin(x))
        elif self.mapping == 'sigmoid':
            return torch.sigmoid(self.lin(x))
        elif self.mapping == 'kernel':
            z = self.lin(x)
            scale = (self.embedding_dim / 2)**-0.5
            return scale * torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        else:
            raise ValueError(f"Unknown mapping: {self.mapping!r}")

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, 1.0, 1.0)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self._transform(self.x[start]).view(pos_rw.size(0), 1,
                                                      self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self._transform(self.x[start]).view(neg_rw.size(0), 1,
                                                      self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(*args, solver=solver,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')
