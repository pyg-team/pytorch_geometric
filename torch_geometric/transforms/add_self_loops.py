from typing import Union

from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops


@functional_transform('add_self_loops')
class AddSelfLoops(BaseTransform):
    r"""Adds self-loops to the given homogeneous or heterogeneous graph
    (functional name: :obj:`add_self_loops`).

    Args:
        attr (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.add_self_loops`.
            (default: :obj:`"edge_weight"`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    """
    def __init__(
        self,
        attr: str = 'edge_weight',
        fill_value: Union[float, Tensor, str] = 1.0,
    ) -> None:
        self.attr = attr
        self.fill_value = fill_value

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store[self.attr] = add_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
                fill_value=self.fill_value,
                num_nodes=store.size(0),
            )

        return data
