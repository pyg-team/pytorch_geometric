import torch
from torch.nn import Embedding
from torch_sparse import SparseTensor
from sklearn.linear_model import LogisticRegression

EPS = 1e-15


class MetaPath2Vec(torch.nn.Module):
    def __init__(self, edge_index_dict, embedding_dim, metapath,
                 walks_per_node=1, num_nodes_dict=None, sparse=False):
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
            adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=sizes)
            adj_dict[keys] = adj

        self.adj_dict = adj_dict
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walks_per_node = walks_per_node
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
        offset += [self.start[x[-1]] for x in metapath]
        self.register_buffer('offset', torch.tensor(offset))

        self.embedding = Embedding(count, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, node_type, subset=None):
        """Returns the embeddings for the nodes in :obj:`subset` of type
        :obj:`node_type`."""
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if subset is None else emb[subset]

    def __positive_sampling__(self, subset):
        device = self.embedding.weight.device

        subsets = []
        for keys in self.metapath:
            adj = self.adj_dict[keys]
            subset = adj.sample(num_neighbors=1, subset=subset).squeeze()
            subsets.append(subset)
        out = torch.stack(subsets, dim=-1).to(device)
        out.add_(self.offset[1:].view(1, -1))
        return out

    def __negative_sampling__(self, subset):
        device = self.embedding.weight.device

        subsets = []
        for keys in self.metapath:
            node_type = keys[-1]
            subset = torch.randint(self.start[node_type], self.end[node_type],
                                   (subset.size(0), ), dtype=torch.long,
                                   device=device)
            subsets.append(subset)
        return torch.stack(subsets, dim=-1)

    def loss(self, subset):
        r"""Computes the loss for the nodes in :obj:`subset` with negative
        sampling."""
        subset = subset.repeat(self.walks_per_node)

        pos_rest = self.__positive_sampling__(subset)
        neg_rest = self.__negative_sampling__(subset)

        start = subset + self.start[self.metapath[0][0]]

        h_start = self.embedding(start)
        h_start = h_start.view(subset.size(0), 1, -1)

        h_pos_rest = self.embedding(pos_rest.view(-1))
        h_pos_rest = h_pos_rest.view(subset.size(0), pos_rest.size(-1), -1)

        h_neg_rest = self.embedding(neg_rest.view(-1))
        h_neg_rest = h_neg_rest.view(subset.size(0), neg_rest.size(-1), -1)

        out = (h_start * h_pos_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        out = (h_start * h_neg_rest).sum(dim=-1).view(-1)
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
                                   self.embedding_dim)
