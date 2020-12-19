from typing import Optional, Dict, Tuple, List
from torch_geometric.typing import Adj

import copy
import random
from itertools import chain, combinations

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor


def powerset(rels):
    arange = range(1, len(rels) + 1)
    return list(chain.from_iterable(combinations(rels, i) for i in arange))


class NARS(torch.nn.Module):
    r"""Thei Neighbor Averaging over Relation Subgraphs (NARS) algorithm, which
    trains a classifier on neighbor-averaged features for randomly-sampled
    subgraphs of relations in a heterogeneous graph

    .. math::
        \mathbf{x}_i,r^{(\ell)} = \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{x}_^{(\ell - 1)}_j,r

    where :math:`r` denotes a set of sampled relations.

    .. note::

        For an example of using NARS, see `examples/nars.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        nars.py>`_.

    Args:
        num_hops (int): The number of hops for propagating messages.
        num_sampled_subsets (int): The number of subsampled relations from
            the original heterogeneous graph.
        num_features (int): The dimensionality of node features.
    """
    def __init__(self, num_hops: int, num_sampled_subsets: int,
                 num_features: int):
        super(NARS, self).__init__()
        self.num_hops = num_hops
        self.num_sampled_subsets = num_sampled_subsets
        self.num_features = num_features

        self.weight = Parameter(
            torch.Tensor(num_hops + 1, num_sampled_subsets, 1, num_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    @torch.no_grad()
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Adj],
        subsets: Optional[List[List[str]]] = None,
    ) -> Dict[str, Tensor]:
        """"""

        # Get the (subsampled) powerset of relations.
        if subsets is None:
            rels = list(edge_index_dict.keys())
            subsets = powerset(rels)
            if self.num_sampled_subsets:
                subsets = random.sample(subsets, k=self.num_sampled_subsets)

        # Convert edge indices to SparseTensor format.
        edge_index_dict = copy.copy(edge_index_dict)
        for key, edge_index in edge_index_dict.items():
            if isinstance(edge_index, Tensor):
                row, col = edge_index
                src, e_type, dst = key
                sparse_sizes = (x_dict[dst].size(0), x_dict[src].size(0))
                adj_t = SparseTensor(row=col, col=row,
                                     sparse_sizes=sparse_sizes)
                edge_index_dict[key] = adj_t

        # Generate outputs: [num_hops + 1, num_subsets, num_nodes, num_feats]
        out_dict = {}
        for key, item in x_dict.items():
            out_dict[key] = torch.zeros(self.num_hops + 1, len(subsets),
                                        *item.size())
            out_dict[key][0] = x_dict[key].unsqueeze(0).cpu()

        for j, subset in enumerate(subsets):
            norm_dict = {
                key: item.new_zeros(item.size(0), dtype=torch.long)
                for key, item in x_dict.items()
            }

            for i in range(1, self.num_hops + 1):
                for (src, e_type, dst) in subset:
                    adj_t = edge_index_dict[(src, e_type, dst)]

                    x_src = out_dict[src][i - 1, j].to(adj_t.device())
                    x_dst = out_dict[dst][i - 1, j].to(adj_t.device())

                    out_dict[dst][i, j] += adj_t.matmul(x_src).cpu()
                    out_dict[src][i, j] += adj_t.t().matmul(x_dst).cpu()

                    if i == 1:
                        norm_dict[dst] += adj_t.storage.rowcount()
                        norm_dict[src] += adj_t.storage.colcount()

                if i == 1:
                    for key, norm in norm_dict.items():
                        norm = 1.0 / norm.to(torch.float)
                        norm[torch.isinf(norm)] = 0.
                        norm_dict[key] = norm.cpu()

                for key, norm in norm_dict.items():
                    out_dict[key][i, j] *= norm.view(-1, 1)

        return out_dict

    def weighted_aggregation(self, x: Tensor) -> Tensor:
        # x [num_hops, num_subsets, num_nodes, num_features]
        return (x * self.weight).sum(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_hops={self.num_hops}, '
                f'num_sampled_subsets={self.num_sampled_subsets}, '
                f'num_features={self.num_features}')
