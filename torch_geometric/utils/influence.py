from typing import List, Tuple, Union, cast

import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def k_hop_subsets_rough(
    node_idx: int,
    num_hops: int,
    edge_index: Tensor,
    num_nodes: int,
) -> List[Tensor]:
    r"""Return *rough* (possibly overlapping) *k*-hop node subsets.

    This is a thin wrapper around
    :pyfunc:`torch_geometric.utils.k_hop_subgraph` that *additionally* returns
    **all** intermediate hop subsets rather than the full union only.

    Parameters
    ----------
    node_idx: int
        Index or indices of the central node(s).
    num_hops: int
        Number of hops *k*.
    edge_index: Tensor
        Edge index in COO format with shape :math:`[2, \text{num_edges}]`.
    num_nodes: int
        Total number of nodes in the graph. Required to allocate the masks.

    Returns:
    -------
    List[Tensor]
        A list ``[H₀, H₁, …, H_k]`` where ``H₀`` contains the seed node(s) and
        ``H_i`` (for *i*>0) contains **all** nodes that are exactly *i* hops
        away in the *expanded* neighbourhood (i.e. overlaps are *not*
        removed).
    """
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx_ = torch.tensor([node_idx], device=row.device)

    subsets = [node_idx_]
    for _ in range(num_hops):
        node_mask.zero_()
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    return subsets


def k_hop_subsets_exact(
    node_idx: int,
    num_hops: int,
    edge_index: Tensor,
    num_nodes: int,
    device: Union[torch.device, str],
) -> List[Tensor]:
    """Return **disjoint** *k*-hop subsets.

    This function refines :pyfunc:`k_hop_subsets_rough` by removing nodes that
    have already appeared in previous hops, ensuring that each subset contains
    nodes *exactly* *i* hops away from the seed.
    """
    rough_subsets = k_hop_subsets_rough(node_idx, num_hops, edge_index,
                                        num_nodes)

    exact_subsets: List[List[int]] = [rough_subsets[0].tolist()]
    visited: set[int] = set(exact_subsets[0])

    for hop_subset in rough_subsets[1:]:
        fresh = set(hop_subset.tolist()) - visited
        visited |= fresh
        exact_subsets.append(list(fresh))

    return [
        torch.tensor(s, device=device, dtype=edge_index.dtype)
        for s in exact_subsets
    ]


def jacobian_l1(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int,
    device: Union[torch.device, str],
    *,
    vectorize: bool = True,
) -> Tensor:
    """Compute the **L1 norm** of the Jacobian for a given node.

    The Jacobian is evaluated w.r.t. the node features of the *k*-hop induced
    sub‑graph centred at ``node_idx``. The result is *folded back* onto the
    **original** node index space so that the returned tensor has length
    ``data.num_nodes``, where the influence score will be zero for nodes
    outside the *k*-hop subgraph.

    Notes:
    -----
    *   The function assumes that the model *and* ``data.x`` share the same
        floating‑point precision (e.g. both ``float32`` or both ``float16``).

    """
    # Build the induced *k*-hop sub‑graph (with node re‑labelling).
    edge_index = cast(Tensor, data.edge_index)
    x = cast(Tensor, data.x)
    k_hop_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, max_hops, edge_index, relabel_nodes=True)
    # get the location of the *center* node inside the sub‑graph
    root_pos = cast(int, mapping[0])

    # Move tensors & model to the correct device
    device = torch.device(device)
    sub_x = x[k_hop_nodes].to(device)
    sub_edge_index = sub_edge_index.to(device)
    model = model.to(device)

    # Jacobian evaluation
    def _forward(x: Tensor) -> Tensor:
        return model(x, sub_edge_index)[root_pos]

    jac = jacobian(_forward, sub_x, vectorize=vectorize)
    influence_sub = jac.abs().sum(dim=(0, 2))  # Sum of L1 norm
    num_nodes = cast(int, data.num_nodes)
    # Scatter the influence scores back to the *global* node space
    influence_full = torch.zeros(num_nodes, dtype=influence_sub.dtype,
                                 device=device)
    influence_full[k_hop_nodes] = influence_sub

    return influence_full


def jacobian_l1_agg_per_hop(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int,
    device: Union[torch.device, str],
    vectorize: bool = True,
) -> Tensor:
    """Aggregate Jacobian L1 norms **per hop** for node_idx.

    Returns a vector ``[I_0, I_1, …, I_k]`` where ``I_i`` is the *total*
    influence exerted by nodes that are exactly *i* hops away from
    ``node_idx``.
    """
    num_nodes = cast(int, data.num_nodes)
    edge_index = cast(Tensor, data.edge_index)
    influence = jacobian_l1(model, data, max_hops, node_idx, device,
                            vectorize=vectorize)
    hop_subsets = k_hop_subsets_exact(node_idx, max_hops, edge_index,
                                      num_nodes, influence.device)
    sigle_node_influence_per_hop = [influence[s].sum() for s in hop_subsets]
    return torch.tensor(sigle_node_influence_per_hop, device=influence.device)


def avg_total_influence(
    influence_all_nodes: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Compute the *influence‑weighted receptive field* ``R``."""
    avg_total_influences = torch.mean(influence_all_nodes, dim=0)
    if normalize:  # nomalize by hop_0 (jacobian of the center node feature)
        avg_total_influences = avg_total_influences / avg_total_influences[0]
    return avg_total_influences


def influence_weighted_receptive_field(T: Tensor) -> float:
    """Compute the *influence‑weighted receptive field* ``R``.

    Given an influence matrix ``T`` of shape ``[N, k+1]`` (i‑th row contains
    the per‑hop influences of node *i*), the receptive field breadth *R* is
    defined as the expected hop distance when weighting by influence.

    A larger *R* indicates that, on average, influence comes from **farther**
    hops.
    """
    normalised = T / torch.sum(T, dim=1, keepdim=True)
    hops = torch.arange(T.shape[1]).float()  # 0 … k
    breadth = normalised @ hops  # shape (N,)
    return breadth.mean().item()


def total_influence(
    model: torch.nn.Module,
    data: Data,
    max_hops: int,
    num_samples: Union[int, None] = None,
    normalize: bool = True,
    average: bool = True,
    device: Union[torch.device, str] = "cpu",
    vectorize: bool = True,
) -> Tuple[Tensor, float]:
    r"""Compute Jacobian‑based influence aggregates for *multiple* seed nodes,
    as introduced in the
    `"Towards Quantifying Long-Range Interactions in Graph Machine Learning:
    a Large Graph Dataset and a Measurement"
    <https://arxiv.org/abs/2503.09008>`_ paper.
    This measurement quantifies how a GNN model's output at a node is
    influenced by features of other nodes at increasing hop distances.

    Specifically, for every sampled node :math:`v`, this method

    1. evaluates the **L1‑norm** of the Jacobian of the model output at
       :math:`v` w.r.t. the node features of its *k*-hop induced sub‑graph;
    2. sums these scores **per hop** to obtain the influence vector
       :math:`(I_{0}, I_{1}, \dots, I_{k})`;
    3. optionally averages those vectors over all sampled nodes and
       optionally normalises them by :math:`I_{0}`.

    Please refer to Section 4 of the paper for a more detailed definition.

    Args:
        model (torch.nn.Module): A PyTorch Geometric‑compatible model with
            forward signature ``model(x, edge_index) -> Tensor``.
        data (torch_geometric.data.Data): Graph data object providing at least
            :obj:`x` (node features) and :obj:`edge_index` (connectivity).
        max_hops (int): Maximum hop distance :math:`k`.
        num_samples (int, optional): Number of random seed nodes to evaluate.
            If :obj:`None`, all nodes are used. (default: :obj:`None`)
        normalize (bool, optional): If :obj:`True`, normalize each hop‑wise
            influence by the influence of hop 0. (default: :obj:`True`)
        average (bool, optional): If :obj:`True`, return the hop‑wise **mean**
            over all seed nodes (shape ``[k+1]``).
            If :obj:`False`, return the full influence matrix of shape
            ``[N, k+1]``. (default: :obj:`True`)
        device (torch.device or str, optional): Device on which to perform the
            computation. (default: :obj:`"cpu"`)
        vectorize (bool, optional): Forwarded to
            :func:`torch.autograd.functional.jacobian`.  Keeping this
            :obj:`True` is often faster but increases memory usage.
            (default: :obj:`True`)

    Returns:
        Tuple[Tensor, float]:
            * **avg_influence** (*Tensor*):
              shape ``[k+1]`` if :obj:`average=True`;
              shape ``[N, k+1]`` otherwise.
            * **R** (*float*): Influence‑weighted receptive‑field breadth
              returned by :func:`influence_weighted_receptive_field`.

    Example::
        >>> avg_I, R = total_influence(model, data, max_hops=3,
        ...                            num_samples=1000)
        >>> avg_I
        tensor([1.0000, 0.1273, 0.0142, 0.0019])
        >>> R
        0.216
    """
    num_samples = data.num_nodes if num_samples is None else num_samples
    num_nodes = cast(int, data.num_nodes)
    nodes = torch.randperm(num_nodes)[:num_samples].tolist()

    influence_all_nodes: List[Tensor] = [
        jacobian_l1_agg_per_hop(model, data, max_hops, n, device,
                                vectorize=vectorize)
        for n in tqdm(nodes, desc="Influence")
    ]
    allnodes = torch.vstack(influence_all_nodes).detach().cpu()

    # Average total influence at each hop
    if average:
        avg_influence = avg_total_influence(allnodes, normalize=normalize)
    else:
        avg_influence = allnodes

    # Influence‑weighted receptive field
    R = influence_weighted_receptive_field(allnodes)

    return avg_influence, R
