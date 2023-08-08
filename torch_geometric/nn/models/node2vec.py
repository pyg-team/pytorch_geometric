from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr, ptr2index

####################################################################################### { define a weighted random walk that care the data.attr
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def random_walk_weighted(rowptr, col, weight, batch, walk_length, p, q):
    row = ptr2index(rowptr) 
    row_repeat = row.repeat(len(batch), 1)
    col_repeat = col.repeat(len(batch), 1)    
    pos_mats = batch.unsqueeze(dim=0)
    loc_mats = batch.unsqueeze(dim=0)
    weights_tmp = weight.repeat(len(batch),1).clone().detach()
    indices = batch.clone().detach()
    for i in range(walk_length):
        starts_with = rowptr[indices]
        ends_with = rowptr[indices+1]

        weights = (weights_tmp).clone().detach()
        weights[ torch.arange(0, len(col), dtype=torch.float32).repeat(len(batch),1).to(device) < starts_with.unsqueeze(1) ] = 0
        weights[ torch.arange(0, len(col), dtype=torch.float32).repeat(len(batch),1).to(device) >= ends_with.unsqueeze(1) ] = 0
        
        ## handling p
        row_nodes = row_repeat.gather(1,loc_mats[-1].unsqueeze(1)).clone().detach()
        col_nodes = col_repeat.gather(1,loc_mats[-1].unsqueeze(1)).clone().detach()
        weights[(row_repeat==col_nodes) & (col_repeat==row_nodes)] = torch.clamp(torch.tensor(1-p),0,1) * weights[(row_repeat==col_nodes) & (col_repeat==row_nodes)]


        pdfs = weights / weights.sum(dim=1).repeat(len(col),1).t()
        cdfs = torch.cumsum(pdfs, 1)
        rnds = torch.rand(len(batch)).to(device)
        locs = torch.max(rnds.unsqueeze(1)<cdfs, dim=1).indices
        loc_mats = torch.cat((loc_mats, locs.unsqueeze(dim=0)),0)
        pos_mats = torch.cat((pos_mats, col[locs].unsqueeze(dim=0)),0)

        ## handling q
        q_rnd = torch.rand(len(batch))
        q_clamp = torch.clamp(torch.tensor(q),0,1).repeat(len(batch))
        indices_tmp = (col[locs]).clone().detach()
        indices[(q_rnd < q_clamp)] = indices_tmp[(q_rnd < q_clamp)]
                 
    directions = pos_mats.t()
    locations = loc_mats[1:].t()
    return(directions, locations)
# ###################################################################################### }


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
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        edge_attr (torch.Tensor): The edge attributes.                                              # Explain
        fast_mode ()bool, optional: If set to :obj:'True', the weights in                           # Explain
            edge_attr are converted to posibilities (where in each epoch it 
            converge to the real weights). Else if set to :obj:'False' the real 
            weights are computed.        
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
    """
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        edge_attr: Tensor = None,                                                    # Add data attribute to the input
        fast_mode: bool = True,                                                       # Fast Estimate or Exact Computation
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
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

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).to(device)
        if edge_attr is None:                                                      # Check if data has edge.attr
            self.edge_weight = torch.ones((1,len(edge_index[0])))[0].to(device)   
        else:
            self.edge_weight = edge_attr.t()[0].to(device)                         
            
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col
        self.row = row.to(device)                                                              # Add row to self

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.fast_mode = fast_mode                                                          # Add parameter
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)

        ########################################################################################## {if fast mode,  replace the weights with posibility. else, use the scrach def
        if self.fast_mode:
            row_all, col_all, weight_all = self.row, self.col, self.edge_weight
            probs = torch.rand(len(weight_all)).to(device)<weight_all
            row = row_all[probs]
            col = col_all[probs]
            weight = weight_all[probs]
            rowptr = index2ptr(row, self.num_nodes)
            rw = self.random_walk_fn(rowptr, col, batch, self.walk_length, self.p, self.q)
        else:
            rw = random_walk_weighted(self.rowptr, self.col, self.edge_weight, batch, self.walk_length, self.p, self.q)
        ########################################################################################## }
        
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
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch).to(device)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
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
        multi_class: str = 'auto',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')
