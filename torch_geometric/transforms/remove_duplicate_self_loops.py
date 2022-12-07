from typing import Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import segregate_self_loops


@functional_transform('remove_duplicate_self_loops')
class RemoveDuplicateSelfLoops(BaseTransform):
    r"""Removes self-loops to the given homogeneous or heterogeneous graph
    (functional name: :obj:`remove_self_loops`).

    Args:

    """
    def __init__(self, keep_original_order: bool = True):
        self.keep_original_order = keep_original_order
        
    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            (edge_index, edge_weight, loop_edge_index, loop_edge_attr) = segregate_self_loops(store.edge_index, getattr(store, 'edge_weight', None))
            
            loop_edge_index = torch.stack((loop_edge_index[0][::2],loop_edge_index[0][::2]))

            edge_index = torch.cat((edge_index, loop_edge_index), dim=1)
            setattr(store, 'edge_index', edge_index)
            
            if edge_weight is not None:
                loop_edge_attr = loop_edge_attr[::2]
                setattr(store, 'edge_weight', edge_weight)
        
        return data
