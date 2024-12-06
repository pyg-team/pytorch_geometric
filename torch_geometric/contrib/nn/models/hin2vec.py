from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Embedding, Sigmoid
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import WITH_TORCH_CLUSTER, EdgeType, NodeType
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict


class HIN2Vec(torch.nn.Module):
    r"""The HIN2Vec model from the `"HIN2Vec: Explore Meta-paths in
    Heterogeneous Information Networks for Representation Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3132847.3132953>`_ paper where node
    embeddings in a heterogeneous graph are learned via a target sets of
    relationships in the form of meta-paths.

    .. note::

        For examples of using HIN2Vec, see
        `examples/contrib/hin2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/hin2vec.py>`_.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Dictionary
            holding edge indices for each
            :obj:`(src_node_type, rel_type, dst_node_type)` edge type present
            in the heterogeneous graph.
        embedding_dim (int): The size of each embedding vector.
        metapath_length (int): The max length of metapath.
        walk_length (int): The length of random walk to sample the relationship
            in meta-path. This should be larger than `metapath_length`.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        reg (str, optional): Name of function to regularize path vector
            to [0, 1]. Two options: 'step' for binary step function or
            'sigmoid' for sigmoid function. (default: 'step')
        num_nodes_dict (Dict[str, int], optional): Dictionary holding the
            number of nodes for each node type. (default: :obj:`None`)
        allow_circle (bool, optional): If set to :obj:`True`, allows circle in
            the metapath, ie. start node equal to end node.
            (default: :obj:`False`)
        same_neg_node_type (bool, optional): If set to :obj:`True`, the
            negative sample has same node type as positive sample.
            (default: :obj:`False`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath_length: int,
        walk_length: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        reg: str = 'step',
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        allow_circle: bool = False,
        same_neg_node_type: bool = False,
        sparse: bool = False,
    ):
        super().__init__()

        if metapath_length < 1:
            raise ValueError("The metapath_length must be positive integer.")

        if walk_length <= metapath_length:
            raise ValueError(
                "The 'walk_length' is not longer than the 'metapath_length.'")

        if reg == 'sigmoid':
            self.reg = Sigmoid()
        elif reg == 'step':
            self.reg = _BinaryStepModule()
        else:
            raise ValueError(
                "Unsupported regularization function. Ensure that it is either"
                " 'step' or 'sigmoid'.")

        if not WITH_TORCH_CLUSTER:
            raise ImportError(f"'{self.__class__.__name__}' "
                              f"requires the 'torch-cluster' package")
        self.random_walk_fn = torch.ops.torch_cluster.random_walk
        self.edge_index_dict = edge_index_dict
        self.embedding_dim = embedding_dim
        self.metapath_length = metapath_length
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.allow_circle = allow_circle
        self.same_neg_node_type = same_neg_node_type
        self.sparse = sparse
        self.EPS = 1e-15
        self.num_nodes_dict = maybe_num_nodes_dict(edge_index_dict,
                                                   num_nodes_dict)

        # Keeps start and end index of each node type in embedding matrix.
        self.node_start, self.node_end = {}, {}
        node_count = 0
        for node_type, num_nodes in sorted(self.num_nodes_dict.items()):
            self.node_start[node_type] = node_count
            node_count += num_nodes
            self.node_end[node_type] = node_count
        self.node_count = node_count

        if self.same_neg_node_type:
            # Each row corresponds to a node and has (node_count, start_idx) of
            # same node type of the node.
            self.node_type_table = torch.zeros((self.node_count, 2),
                                               dtype=torch.long)
            for node_type in self.node_start:
                start_idx = self.node_start[node_type]
                end_idx = self.node_end[node_type]
                self.node_type_table[start_idx:end_idx,
                                     0] = end_idx - start_idx
                self.node_type_table[start_idx:end_idx, 1] = start_idx

        self.edge_type_id_dict = {}
        self.rowptr, self.col, self.edge_id = self._process_edge_index_dict(
            edge_index_dict)

        # Generate random walks and get distinct metapaths in the walk.
        self.rw = self._generate_random_walk()
        self.metapath_dict = self._extract_distinct_metapath(self.rw[1])

        # Start index of each w length path in path_embedding tensor
        self.path_start = {}
        path_count = 0
        for w in range(1, self.metapath_length + 1):
            self.path_start[w] = path_count
            path_count += self.metapath_dict[w].size(0)

        self.node_embedding = Embedding(self.node_count, embedding_dim,
                                        sparse=sparse)
        self.path_embedding = Embedding(path_count, embedding_dim,
                                        sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_embedding.reset_parameters()
        self.path_embedding.reset_parameters()

    def forward(self, node_type: str,
                batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type` and metapath embeddings.
        """
        start, end = self.node_start[node_type], self.node_end[node_type]
        node_emb = self.node_embedding.weight[start:end]
        if batch is not None:
            node_emb = node_emb[batch]
        return (node_emb, self.path_embedding.weight)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph.

        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.node_count * self.walks_per_node),
                          collate_fn=self._sample, **kwargs)

    def _sample(self, batch: Union[List[int],
                                   Tensor]) -> Tuple[Tensor, Tensor]:
        r"""Generate positive and negative samples within self.metapath_length
        from the random walk with each row of sample of form
        (start_node end_node metapath_id).
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)

        node_seq, edge_seq = self.rw[0][batch], self.rw[1][batch]

        pos = []
        for w in range(self.metapath_length):
            for i in range(edge_seq.size(1) - w):
                metapath = edge_seq[:, i:i + w + 1]
                mask = metapath >= 0  # Filter end of walk
                mask = mask.all(1)
                metapath = metapath[mask, :]
                # Match metapath to id
                metapath_id = (metapath.unsqueeze(1) == self.metapath_dict[w +
                                                                           1]
                               ).all(-1).nonzero()[:, -1]
                assert metapath_id.size(0) == metapath.size(0)
                metapath_id += self.path_start[w + 1]

                start_nodes = node_seq[:, i][mask]
                end_nodes = node_seq[:, i + w + 1][mask]
                pos.append(
                    torch.stack((start_nodes, end_nodes, metapath_id), dim=-1))
        pos_sample = torch.cat(pos, dim=0)
        if not self.allow_circle:
            pos_sample = self._remove_circle(pos_sample)

        # Corrupt the tail to generate negative sample
        neg_sample = pos_sample.repeat(self.num_negative_samples, 1)
        if self.same_neg_node_type:
            neg_node_idx = torch.rand((neg_sample.size(0), ))
            # Multiply with count of same node type as positive sample
            neg_node_idx = torch.floor(
                neg_node_idx * self.node_type_table[neg_sample[:, 1], 0])
            neg_node_idx += self.node_type_table[neg_sample[:, 1], 1]
            neg_sample[:, 1] = neg_node_idx.long()
        else:
            neg_sample[:, 1] = torch.randint(0, self.node_count - 1,
                                             (neg_sample.size(0), ))
        if not self.allow_circle:
            neg_sample = self._remove_circle(neg_sample)

        return pos_sample, neg_sample

    def _remove_circle(self, sample: Tensor) -> Tensor:
        r"""Remove the circle whereby start_node is equal to end_node in sample
        with row (start_node end_node metapath_id).
        """
        mask = sample[:, 0] != sample[:, 1]
        return sample[mask, :]

    def loss(self, pos: Tensor, neg: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative data."""
        # Positive loss
        h_start = self.node_embedding(pos[:, 0])
        h_end = self.node_embedding(pos[:, 1])
        h_path = self.path_embedding(pos[:, 2])
        h_path_reg = self.reg(h_path)
        out = torch.sigmoid((h_start * h_end * h_path_reg).sum(dim=-1))
        pos_loss = -torch.log(out + self.EPS).mean()

        # Negative loss
        h_start = self.node_embedding(neg[:, 0])
        h_end = self.node_embedding(neg[:, 1])
        h_path = self.path_embedding(neg[:, 2])
        h_path_reg = self.reg(h_path)
        out = torch.sigmoid((h_start * h_end * h_path_reg).sum(dim=-1))
        neg_loss = -torch.log(1 - out + self.EPS).mean()

        return pos_loss + neg_loss

    def _process_edge_index_dict(
        self, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Unwrap :obj:`edge_index_dict` into :obj:`(src_node, dst_node,
        edge_type)` tuples. Both `src_node` and `dst_node` are index assigned
        to node in self.node_embedding.
        """
        all_edge_index = []
        for keys, edge_index in edge_index_dict.items():
            if keys not in self.edge_type_id_dict:
                self.edge_type_id_dict[keys] = len(self.edge_type_id_dict)
            if edge_index.size(1) == 0:
                continue
            edge_index_with_offset = edge_index.clone()
            edge_index_with_offset[0] += self.node_start[keys[0]]
            edge_index_with_offset[1] += self.node_start[keys[-1]]
            edge_id = torch.full((1, edge_index.size(1)),
                                 self.edge_type_id_dict[keys],
                                 device=edge_index.device)
            edge_index_with_offset = torch.cat(
                (edge_index_with_offset, edge_id), dim=0)
            all_edge_index.append(edge_index_with_offset)

        all_edge_index = torch.cat(all_edge_index, dim=1)
        edge_index, edge_id = sort_edge_index(all_edge_index[:2],
                                              all_edge_index[2],
                                              self.node_count)
        row, col = edge_index.cpu()
        rowptr = index2ptr(row, self.node_count)
        edge_id = edge_id.cpu()
        return rowptr, col, edge_id

    def _generate_random_walk(self) -> Tuple[Tensor, Tensor]:
        r"""Generate random walks for all the nodes."""
        start_nodes = torch.tensor(range(self.node_count)).repeat(
            self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, start_nodes,
                                 self.walk_length, 1.0, 1.0)
        node_seq, edge_seq = rw

        # -1 indicates no edge in random walk
        mask = edge_seq < 0
        edge_seq = edge_seq.clamp(min=0, max=len(self.edge_id) - 1)

        edge_id_seq = self.edge_id[edge_seq]
        edge_id_seq[mask] = -1
        return node_seq, edge_id_seq

    def _extract_distinct_metapath(self, rw: Tensor) -> Dict[int, Tensor]:
        r"""Extract distinct metapath within self.metapath_length hops from
        each random walk. Each metapath is a sequence of edge types.
        """
        distinct_path_dict = {}
        for w in range(self.metapath_length):
            metapath_w = []
            for i in range(rw.size(1) - w):
                metapath = rw[:, i:i + w + 1]
                # Filter out row with -1 value which indicates end of random
                # walk due to no next edge.
                mask = metapath >= 0
                mask = mask.all(1)
                metapath = metapath[mask, :]
                metapath = torch.unique(metapath, dim=0)
                metapath_w.append(metapath)

            unique_metapath_w = torch.unique(torch.cat(metapath_w), dim=0)

            distinct_path_dict[w + 1] = unique_metapath_w
        return distinct_path_dict

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
                f'{self.node_embedding.weight.size(0)}, '
                f'{self.node_embedding.weight.size(1)})')


class _BinaryStepFunction(Function):
    @staticmethod
    def forward(input: Tensor) -> Tensor:
        return (input > 0).float()

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any):
        input, = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        input, = ctx.saved_tensors
        dx = input.clone()
        mask = (-6 <= dx) & (dx <= 6)
        dx[mask] = torch.sigmoid(dx[mask]) * (1 - torch.sigmoid(dx[mask]))
        return grad_output * dx


class _BinaryStepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return _BinaryStepFunction.apply(input)
