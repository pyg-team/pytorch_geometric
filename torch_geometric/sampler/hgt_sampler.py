from typing import Dict, List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.utils import remap_keys, to_hetero_csc
from torch_geometric.typing import (
    WITH_TORCH_SPARSE,
    EdgeType,
    NodeType,
    OptTensor,
)


class HGTSampler(BaseSampler):
    r"""An implementation of an in-memory heterogeneous layer-wise sampler
    user by :class:`~torch_geometric.loader.HGTLoader`.
    """
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(
                f"'{self.__class__.__name__}' requires 'torch-sparse'")

        if isinstance(data, Data) or isinstance(data, tuple):
            raise NotImplementedError(
                f'{self.__class__.__name__} does not support a data object of '
                f'type {type(data)}.')

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in data.node_types}

        self.node_types, self.edge_types = data.metadata()
        self.num_samples = num_samples
        self.num_hops = max([len(v) for v in num_samples.values()])

        # Conversion to/from C++ string type (see `NeighborSampler`):
        self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
        self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}

        # Convert the graph data into a suitable format for sampling:
        colptr_dict, row_dict, self.perm = to_hetero_csc(
            data, device='cpu', share_memory=share_memory, is_sorted=is_sorted)
        self.row_dict = remap_keys(row_dict, self.to_rel_type)
        self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> HeteroSamplerOutput:

        node, row, col, edge = torch.ops.torch_sparse.hgt_sample(
            self.colptr_dict,
            self.row_dict,
            {inputs.input_type: inputs.node},
            self.num_samples,
            self.num_hops,
        )

        return HeteroSamplerOutput(
            node=node,
            row=remap_keys(row, self.to_edge_type),
            col=remap_keys(col, self.to_edge_type),
            edge=remap_keys(edge, self.to_edge_type),
            batch=None,
            metadata=(inputs.input_id, inputs.time),
        )

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        pass

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        return self.perm
