from typing import Any, Dict, List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
)
from torch_geometric.sampler.utils import remap_keys, to_hetero_csc
from torch_geometric.typing import EdgeType, NodeType, OptTensor


class HGTSampler(BaseSampler):
    r"""An implementation of an in-memory HGT sampler."""
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        input_type: Optional[Any] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        if isinstance(data, Data) or isinstance(data, tuple):
            raise NotImplementedError(
                f'{self.__class__.__name__} does not support a data object of '
                f'type {type(data)}.')

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in data.node_types}

        self.node_types, self.edge_types = data.metadata()
        self.num_samples = num_samples
        self.num_hops = max([len(v) for v in num_samples.values()])

        assert input_type is not None
        self.input_type = input_type

        # Obtain CSC representations for in-memory sampling:
        out = to_hetero_csc(data, device='cpu', share_memory=share_memory,
                            is_sorted=is_sorted)
        colptr_dict, row_dict, perm_dict = out

        # Conversions to/from C++ string type:
        # Since C++ cannot take dictionaries with tuples as key as input,
        # edge type triplets need to be converted into single strings. This
        # is done by maintaining the following mappings:
        self.to_rel_type = {key: '__'.join(key) for key in self.edge_types}
        self.to_edge_type = {'__'.join(key): key for key in self.edge_types}

        self.row_dict = remap_keys(row_dict, self.to_rel_type)
        self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)
        self.perm = perm_dict

    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> HeteroSamplerOutput:
        input_node_dict = {self.input_type: torch.tensor(index)}
        sample_fn = torch.ops.torch_sparse.hgt_sample
        out = sample_fn(
            self.colptr_dict,
            self.row_dict,
            input_node_dict,
            self.num_samples,
            self.num_hops,
        )
        node, row, col, edge, batch = out + (None, )
        return HeteroSamplerOutput(
            node=node,
            row=remap_keys(row, self.to_edge_type),
            col=remap_keys(col, self.to_edge_type),
            edge=remap_keys(edge, self.to_edge_type),
            batch=batch,
            metadata=len(index),
        )

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        **kwargs,
    ) -> HeteroSamplerOutput:
        # TODO(manan): implement
        raise NotImplementedError

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        return self.perm
