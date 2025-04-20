from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.init import uniform_
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int, optional): The actual context size which is
            considered for positive samples. This parameter increases
            the effective sampling rate by reusing samples across
            different source nodes.`window_size` can be used instead;
            this design ensures backward compatibility with
            older Node2Vec versions.
            (default: :obj:`None`)
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
        num_negative_power (float, optional): Exponent for distorting
            node degree distribution in negative sampling.
            (default: :obj:`0.75`)
        max_norm (float, optional): Maximum norm for embedding vectors,
            if specified.
            (default: :obj:`None`)
        window_size (int, optional): Number of context nodes around a center
            node. Use instead of `context_size` or ensure consistency
            (`window_size = context_size - 1`).
            Requires `walk_length > window_size`.
            (default: :obj:`None`)
        restrict_to_forward_context (bool, optional): If :obj:`True`,only
            sample context nodes *after* the center node (asymmetric window).
            Otherwise, sample from both before and after (symmetric window).
            The default (:obj:`True`) matches the strategy of older Node2Vec
            versions. (default: :obj:`True`)
    """

    def __init__(
            self,
            edge_index: Tensor,
            embedding_dim: int,
            walk_length: int,
            context_size: Optional[int] = None,
            walks_per_node: int = 1,
            p: float = 1.0,
            q: float = 1.0,
            num_negative_samples: int = 1,
            num_nodes: Optional[int] = None,
            sparse: bool = False,
            num_negative_power: float = 0.75,
            max_norm: Optional[float] = None,
            window_size: Optional[int] = None,
            restrict_to_forward_context: bool = True,
    ):
        super().__init__()

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires either the 'pyg-lib' or "
                                  f"'torch-cluster' package")
            else:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires the 'torch-cluster' package")

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        if (context_size is not None) and (window_size is None):
            self.window_size = context_size - 1
        elif (context_size is None) and (window_size is not None):
            self.window_size = window_size
        elif (context_size is not None) and (window_size is not None):
            assert window_size == context_size - 1
            self.window_size = window_size
        else:
            error_message = (
                f"Invalid combination for context_size and window_size. "
                f"Received context_size={context_size}, "
                f"window_size={window_size}. "
                f"Please ensure: "
                f"1. At least one of context_size or window_size is not None."
                f"2. If both are specified, "
                f"window_size must equal context_size - 1."
            )
            raise ValueError(error_message)
        assert walk_length >= self.window_size+1

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1

        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.restrict_to_forward_context = restrict_to_forward_context
        self.output_embedding = Embedding(self.num_nodes, embedding_dim,
                                          sparse=sparse, max_norm=max_norm)
        self.context_embedding = Embedding(self.num_nodes, self.embedding_dim,
                                           sparse=sparse, max_norm=max_norm)
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        # 计算带失真因子的采样权重
        self.num_negative_power = num_negative_power
        sampling_weights = degrees.pow(self.num_negative_power)
        # 归一化成概率分布
        sampling_weights /= sampling_weights.sum()
        # 注册为 buffer，它随模型移动 (CPU/GPU)，但不是可训练参数
        self.register_buffer("sampling_weights", sampling_weights)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform_(self.output_embedding.weight,
                 - 0.5 / self.embedding_dim,
                 0.5 / self.embedding_dim)
        uniform_(self.context_embedding.weight,
                 - 0.5 / self.embedding_dim,
                 0.5 / self.embedding_dim)

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.output_embedding.weight
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        pos_sample_cores = []
        walks = []
        for j in range(1 + self.walk_length - self.window_size):
            pos_sample_cores.append(rw[:, j])
            walks.append(rw[:, j + 1:j + 1 + self.window_size])
        if not self.restrict_to_forward_context:
            for j in range(self.window_size, self.walk_length + 1):
                pos_sample_cores.append(rw[:, j])
                walks.append(rw[:, j - self.window_size:j])
        assert len(pos_sample_cores) != 0
        pos_sample_cores = torch.cat(pos_sample_cores, dim=0).unsqueeze(-1)
        walks = torch.cat(walks, dim=0)
        pos_sample = torch.cat([pos_sample_cores, walks], dim=1)

        neg_sample_cores = pos_sample_cores.repeat_interleave(
            self.num_negative_samples,
            dim=0
        )
        neg_sample = torch.multinomial(
            self.sampling_weights,
            neg_sample_cores.size(0) * self.window_size,
            replacement=True
        ).view(neg_sample_cores.size(0), -1).to(batch.device)
        neg_sample = torch.cat(
            [neg_sample_cores.view(-1, 1), neg_sample],
            dim=-1
        )
        return pos_sample, neg_sample

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.output_embedding(start).view(pos_rw.size(0), 1,
                                                    self.embedding_dim)
        h_rest = self.context_embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                            self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = self.output_embedding(start).view(neg_rw.size(0), 1,
                                                    self.embedding_dim)
        h_rest = self.context_embedding(rest.view(-1)).view(neg_rw.size(0), -1,
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

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.output_embedding.weight.size(0)}, '
                f'{self.output_embedding.weight.size(1)})')
