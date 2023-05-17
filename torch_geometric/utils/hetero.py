from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.nn import ParameterDict

from torch_geometric.typing import Adj, EdgeType, NodeType, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict


def learn_sklearn_heuristic():
    import os
    import time

    from torch_geometric.nn.dense import HeteroLinear, Linear
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    fused_times = {}
    loop_times = {}
    try:
        for num_nodes_per_type in [10**2, 10**3, 10**4, 10**5]:
            for out_feats in [2, 4, 8, 16, 32, 64, 128, 256]:
                for n_feats in [4, 8, 16, 32, 64, 128, 256, 512]:
                    for num_types in [4, 8, 16, 32, 64, 128, 256, 512]:
                        try:
                            if n_feats < out_feats:
                                continue
                            print("benchmarking", num_types, "types w/",
                                  num_nodes_per_type, "nodes per type and",
                                  n_feats, "input features and", out_feats,
                                  "outuput feats")
                            x_dict = {
                                'v' + str(i): torch.randn(
                                    (num_nodes_per_type, n_feats)).cuda()
                                for i in range(num_types)
                            }
                            x = torch.cat(list(x_dict.values()), dim=0)
                            node_type = torch.cat([
                                (j * torch.ones(x_j.shape[0])).long()
                                for j, x_j in enumerate(x_dict.values())
                            ]).cuda()
                            lin = Linear(n_feats, out_feats).cuda()
                            heterolin = HeteroLinear(n_feats, out_feats,
                                                     len(list(x_dict.keys())),
                                                     True).cuda()
                            for i in range(60):
                                if i == 10:
                                    since = time.time()
                                heterolin(x=x, type_vec=node_type)
                            key = (num_types, num_nodes_per_type, n_feats,
                                   out_feats)
                            fused_times[key] = ((time.time() - since) / 50.0)
                            print("Avg time for fuse based=", fused_times[key])
                            for i in range(60):
                                if i == 10:
                                    since = time.time()
                                o = x.new_empty(x.size(0), out_feats)
                                for j in range(num_types):
                                    mask = j == node_type
                                    o[mask] = lin(x[mask])
                            loop_times[key] = ((time.time() - since) / 50.0)
                            print("Avg time for for-loop=", loop_times[key])
                        except:  # noqa
                            continue
    except:  # noqa
        pass
    import numpy as np
    X = np.zeros((len(loop_times), 4))
    y = np.zeros(len(loop_times))
    for i, key in enumerate(loop_times.keys()):
        X[i, :] = key
        loop_time, fused_time = loop_times[key], fused_times[key]
        y[i] = int(fused_time <= loop_time)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    scaler = StandardScaler()
    svm = LinearSVC()
    clf = make_pipeline(scaler, svm)
    clf.fit(X, y)

    print("scaler mean=", scaler.mean_)
    print("scaler scale=", scaler.scale_)
    print("svm weights=", svm.coef_)
    print("svm bias=", svm.intercept_)
    # results on A100:
    # scaler mean=
    #     [  125.11603189 12133.21523472   163.81222321    32.43755536]
    # scaler scale=
    #     [  163.34480422 27572.94543809   177.6426489     56.82103934]
    # svm weights=
    #     [[ 2.43877659e+00  1.67583047e+00 -5.20527282e-04  3.43925501e-01]]
    # svm bias=
    #     [1.20236999]


def segmatmul_heuristic(inputs: Tensor, type_ptr, weight: Tensor):
    num_types = len(type_ptr) - 1
    max_num_nodes_per_types = (type_ptr[1:] - type_ptr[:-1]).max()
    in_feat = inputs.size(1)
    out_feat = weight.size(-1)
    # this heuristic was learned with learn_sklearn_heuristic on an A100
    x = torch.tensor([
        int(num_types),
        int(max_num_nodes_per_types),
        int(in_feat),
        int(out_feat)
    ])
    scale_mean = torch.tensor(
        [125.11603189, 12133.21523472, 163.81222321, 32.43755536])
    scale_scale = torch.tensor(
        [163.34480422, 27572.94543809, 177.6426489, 56.82103934])
    svm_weights = torch.tensor(
        [2.43877659e+00, 1.67583047e+00, -5.20527282e-04, 3.43925501e-01])
    bias = 1.20236999
    x = (x - scale_mean) / scale_scale
    return int(x.dot(svm_weights) >= bias)


def group_hetero_graph(edge_index_dict, num_nodes_dict=None):
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict, num_nodes_dict)

    tmp = list(edge_index_dict.values())[0]

    key2int = {}

    cumsum, offset = 0, {}  # Helper data.
    node_types, local_node_indices = [], []
    local2global = {}
    for i, (key, N) in enumerate(num_nodes_dict.items()):
        key2int[key] = i
        node_types.append(tmp.new_full((N, ), i))
        local_node_indices.append(torch.arange(N, device=tmp.device))
        offset[key] = cumsum
        local2global[key] = local_node_indices[-1] + cumsum
        local2global[i] = local2global[key]
        cumsum += N

    node_type = torch.cat(node_types, dim=0)
    local_node_idx = torch.cat(local_node_indices, dim=0)

    edge_indices, edge_types = [], []
    for i, (keys, edge_index) in enumerate(edge_index_dict.items()):
        key2int[keys] = i
        inc = torch.tensor([offset[keys[0]], offset[keys[-1]]]).view(2, 1)
        edge_indices.append(edge_index + inc.to(tmp.device))
        edge_types.append(tmp.new_full((edge_index.size(1), ), i))

    edge_index = torch.cat(edge_indices, dim=-1)
    edge_type = torch.cat(edge_types, dim=0)

    return (edge_index, edge_type, node_type, local_node_idx, local2global,
            key2int)


def get_unused_node_types(node_types: List[NodeType],
                          edge_types: List[EdgeType]) -> Set[NodeType]:
    dst_node_types = set(edge_type[-1] for edge_type in edge_types)
    return set(node_types) - set(dst_node_types)


def check_add_self_loops(module: torch.nn.Module, edge_types: List[EdgeType]):
    is_bipartite = any([key[0] != key[-1] for key in edge_types])
    if is_bipartite and getattr(module, 'add_self_loops', False):
        raise ValueError(
            f"'add_self_loops' attribute set to 'True' on module '{module}' "
            f"for use with edge type(s) '{edge_types}'. This will lead to "
            f"incorrect message passing results.")


def construct_bipartite_edge_index(
    edge_index_dict: Dict[EdgeType, Adj],
    src_offset_dict: Dict[EdgeType, int],
    dst_offset_dict: Dict[NodeType, int],
    edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
) -> Tuple[Adj, Optional[Tensor]]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding graph connectivity information for each
            individual edge type, either as a :class:`torch.Tensor` of
            shape :obj:`[2, num_edges]` or a
            :class:`torch_sparse.SparseTensor`.
        src_offset_dict (Dict[Tuple[str, str, str], int]): A dictionary of
            offsets to apply to the source node type for each edge type.
        dst_offset_dict (Dict[str, int]): A dictionary of offsets to apply for
            destination node types.
        edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding edge features for each individual edge type.
            (default: :obj:`None`)
    """
    is_sparse_tensor = False
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        dst_offset = dst_offset_dict[edge_type[-1]]

        # TODO Add support for SparseTensor w/o converting.
        is_sparse_tensor = isinstance(edge_index, SparseTensor)
        if is_sparse(edge_index):
            edge_index, _ = to_edge_index(edge_index)
            edge_index = edge_index.flip([0])
        else:
            edge_index = edge_index.clone()

        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_indices.append(edge_index)

        if edge_attr_dict is not None:
            if isinstance(edge_attr_dict, ParameterDict):
                edge_attr = edge_attr_dict['__'.join(edge_type)]
            else:
                edge_attr = edge_attr_dict[edge_type]
            if edge_attr.size(0) != edge_index.size(1):
                edge_attr = edge_attr.expand(edge_index.size(1), -1)
            edge_attrs.append(edge_attr)

    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None
    if edge_attr_dict is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)

    if is_sparse_tensor:
        # TODO Add support for `SparseTensor.sparse_sizes()`.
        edge_index = SparseTensor(
            row=edge_index[1],
            col=edge_index[0],
            value=edge_attr,
        )

    return edge_index, edge_attr
