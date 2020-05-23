import torch
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from sklearn.linear_model import LogisticRegression

EPS = 1e-15


class MetaPath2Vec(torch.nn.Module):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based
    on a given :obj:`metapath` are sampled in a heterogeneous graph, and node
    embeddings are learned via negative sampling optimization.

    .. note::

        For an example of using MetaPath2Vec, see `examples/metapath2vec.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        metapath2vec.py>`_.

    Args:
        edge_index_dict (dict): Dictionary holding edge indices for each
            :obj:`(source_node_type, relation_type, target_node_type)` present
            in the heterogeneous graph.
        embedding_dim (int): The size of each embedding vector.
        metapath (list): The metapath described as a list of
            :obj:`(source_node_type, relation_type, target_node_type)` tuples.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes_dict (dict, optional): Dictionary holding the number of nodes
            for each node type. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """
    def __init__(self, edge_index_dict, embedding_dim, metapath, walk_length,
                 context_size, walks_per_node=1, num_negative_samples=1,
                 num_nodes_dict=None, sparse=False):
        super(MetaPath2Vec, self).__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        adj_dict = {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
            adj = adj.to('cpu')
            adj_dict[keys] = adj

        assert metapath[0][0] == metapath[-1][-1]
        assert walk_length >= context_size

        self.adj_dict = adj_dict
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        self.embedding = Embedding(count, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, node_type, batch=None):
        """Returns the embeddings for the nodes in :obj:`subset` of type
        :obj:`node_type`."""
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs):
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch):
        # device = self.embedding.weight.device

        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = adj.sample(num_neighbors=1, subset=batch).squeeze()
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.embedding.weight.size(0),
                                   self.embedding.weight.size(1))
