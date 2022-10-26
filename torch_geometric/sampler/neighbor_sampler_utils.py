from typing import Callable, Optional, Tuple, Union

import torch

from torch_geometric.sampler.base import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.utils import add_negative_samples

# Node-based sampling #########################################################


def node_sample(
    index: NodeSamplerInput,
    sample_fn: Callable,
    input_type: Optional[str] = None,
    **kwargs,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from a node sampler input, leveraging a sampling
    function that accepts a seed and (optionally) a seed time / seed time
    dictionary as input. Returns the output of this sampling procedure."""
    index, input_nodes, input_time = index

    if input_type is not None:
        # Heterogeneous sampling:
        seed_time_dict = None
        if input_time is not None:
            seed_time_dict = {input_type: input_time}
        output = sample_fn(seed={input_type: input_nodes},
                           seed_time_dict=seed_time_dict)
        output.metadata = index

    else:
        # Homogeneous sampling:
        output = sample_fn(seed=input_nodes, seed_time=input_time)
        output.metadata = index

    return output


# Edge-based sampling #########################################################


def edge_sample(
    index: EdgeSamplerInput,
    sample_fn: Callable,
    num_src_nodes: int,
    num_dst_nodes: int,
    disjoint: bool,
    input_type: Optional[Tuple[str, str]] = None,
    **kwargs,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`."""
    index, row, col, edge_label, edge_label_time = index
    edge_label_index = torch.stack([row, col], dim=0)
    negative_sampling_ratio = kwargs.get('negative_sampling_ratio', 0.0)

    out = add_negative_samples(
        edge_label_index,
        edge_label,
        edge_label_time,
        num_src_nodes,
        num_dst_nodes,
        negative_sampling_ratio,
    )
    edge_label_index, edge_label, edge_label_time = out
    num_seed_edges = edge_label_index.size(1)
    seed_time = seed_time_dict = None

    if input_type is not None:
        # Heterogeneous sampling:
        if input_type[0] != input_type[-1]:
            if disjoint:
                seed_src = edge_label_index[0]
                seed_dst = edge_label_index[1]
                edge_label_index = torch.arange(0,
                                                num_seed_edges).repeat(2).view(
                                                    2, -1)
                seed_dict = {
                    input_type[0]: seed_src,
                    input_type[-1]: seed_dst,
                }
                if edge_label_time is not None:
                    seed_time_dict = {
                        input_type[0]: edge_label_time,
                        input_type[-1]: edge_label_time
                    }
            else:
                seed_src = edge_label_index[0]
                seed_src, inverse_src = seed_src.unique(return_inverse=True)
                seed_dst = edge_label_index[1]
                seed_dst, inverse_dst = seed_dst.unique(return_inverse=True)
                edge_label_index = torch.stack([
                    inverse_src,
                    inverse_dst,
                ], dim=0)
                seed_dict = {
                    input_type[0]: seed_src,
                    input_type[-1]: seed_dst,
                }

        else:  # Merge both source and destination node indices:
            if disjoint:
                seed_nodes = edge_label_index.view(-1)
                edge_label_index = torch.arange(0, 2 * num_seed_edges)
                edge_label_index = edge_label_index.view(2, -1)
                seed_dict = {input_type[0]: seed_nodes}
                if edge_label_time is not None:
                    tmp = torch.cat([edge_label_time, edge_label_time])
                    seed_time_dict = {input_type[0]: tmp}
            else:
                seed_nodes = edge_label_index.view(-1)
                seed_nodes, inverse = seed_nodes.unique(return_inverse=True)
                edge_label_index = inverse.view(2, -1)
                seed_dict = {input_type[0]: seed_nodes}

        output = sample_fn(
            seed=seed_dict,
            seed_time_dict=seed_time_dict,
        )

        if disjoint:
            for key, batch in output.batch.items():
                output.batch[key] = batch % num_seed_edges

        output.metadata = (index, edge_label_index, edge_label,
                           edge_label_time)

    else:
        # Homogeneous sampling:
        if disjoint:
            seed_nodes = edge_label_index.view(-1)
            edge_label_index = torch.arange(0, 2 * num_seed_edges)
            edge_label_index = edge_label_index.view(2, -1)
            if edge_label_time is not None:
                seed_time = torch.cat([edge_label_time, edge_label_time])

        else:
            seed_nodes = edge_label_index.view(-1)
            seed_nodes, inverse = seed_nodes.unique(return_inverse=True)
            edge_label_index = inverse.view(2, -1)

        output = sample_fn(seed=seed_nodes, seed_time=seed_time)

        if disjoint:
            output.batch = output.batch % num_seed_edges

        output.metadata = (index, edge_label_index, edge_label,
                           edge_label_time)

    return output
