import copy
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear, Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import is_sparse, to_edge_index


class FastFiLMConv(MessagePassing):
    r"""See :class:`FiLMConv`.
    Main difference is parrallelizing linear layers at the cost of more memory usage """
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int = 1,
            nn: Optional[Callable] = None,
            act: Optional[Callable] = ReLU(),
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = max(num_relations, 1)
        self.act = act
        self.nn_is_none = nn is None
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.num_relations > 1:
            self.lins = HeteroLinear(in_channels[0], out_channels,
                                     num_types=num_relations, is_sorted=True,
                                     bias=False)
            if self.nn_is_none:
                self.films = HeteroLinear(in_channels[1], 2 * out_channels,
                                          num_types=num_relations,
                                          is_sorted=True)
            else:
                self.films = ModuleList()
                for _ in range(num_relations):
                    self.films.append(copy.deepcopy(nn))
        else:
            self.lins = (Linear(in_channels[0], out_channels, bias=False))
            if self.nn_is_none:
                self.films = Linear(in_channels[1], 2 * out_channels)
            else:
                self.films = copy.deepcopy(nn)

        self.lin_skip = Linear(in_channels[1], self.out_channels, bias=False)
        if self.nn_is_none:
            self.film_skip = Linear(in_channels[1], 2 * self.out_channels,
                                    bias=False)
        else:
            self.film_skip = copy.deepcopy(nn)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lins.reset_parameters()
        if self.nn_is_none:
            self.films.reset_parameters()
        else:
            for f in self.films:
                reset(f)
        self.lin_skip.reset_parameters()
        reset(self.film_skip)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_type: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # need to clone edge_index before incrementing it
        edge_index = edge_index.clone()
        beta, gamma = self.film_skip(x[1]).split(self.out_channels, dim=-1)
        out = gamma * self.lin_skip(x[1]) + beta
        if self.act is not None:
            out = self.act(out)
        # propagate_type: (x: Tensor, beta: Tensor, gamma: Tensor)
        if self.num_relations <= 1:
            beta, gamma = self.films(x[1]).split(self.out_channels, dim=-1)
            out = out + self.propagate(edge_index, x=self.lins(x[0]),
                                       beta=beta, gamma=gamma, size=None)
        else:
            # (TODO) add support for sparse tensors without conversion
            if is_sparse(edge_index):
                print(
                    "Warning: sparse edge representations are not supported for FastFiLMConv yet.\
                       This incurs an additional conversion each forward pass."
                )
                edge_index = to_edge_index(edge_index)[0]
            film_xs, propogate_xs, type_list_films, type_list_lins = [], [], [], []
            prop_count, film_count = 0, 0
            film_x = x[1]
            prop_x = x[0]
            # duplicate xs and increment edge indices accordingly
            for e_type_i in range(self.num_relations):
                film_xs.append(film_x)
                propogate_xs.append(prop_x)
                type_list_films.append(
                    torch.full((film_x.size(0), ), e_type_i, dtype=torch.long))
                type_list_lins.append(
                    torch.full((prop_x.size(0), ), e_type_i, dtype=torch.long))
                mask = edge_type == e_type_i
                edge_index[0, mask] += prop_count
                edge_index[1, mask] += film_count
                prop_count += prop_x.size(0)
                film_count += film_x.size(0)
            # cat and apply linears
            if self.nn_is_none:
                beta, gamma = self.films(torch.cat(film_xs),
                                         torch.cat(type_list_films)).split(
                                             self.out_channels, dim=-1)
            else:
                beta, gamma = torch.cat([
                    self.films[i](film_x) for i, film_x in enumerate(film_xs)
                ]).split(self.out_channels, dim=-1)
            propogate_x = self.lins(torch.cat(propogate_xs),
                                    torch.cat(type_list_lins))

            # propogate
            out += sum(
                self.propagate(
                    edge_index, x=propogate_x, beta=beta, gamma=gamma,
                    size=None).split(
                        int(propogate_x.size(0) / self.num_relations), dim=0))

        return out

    def message(self, x_j: Tensor, beta_i: Tensor, gamma_i: Tensor) -> Tensor:
        out = gamma_i * x_j + beta_i
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
