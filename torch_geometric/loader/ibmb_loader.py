import logging
import math
from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import get_ppr, is_undirected, subgraph

try:
    import numba
    WITH_NUMBA = True
except ImportError:  # pragma: no cover
    WITH_NUMBA = False


class OutputNodes(NamedTuple):
    seed_id: Tensor
    auxiliary_id: Tensor


class _IBMBBaseLoader(torch.utils.data.DataLoader):
    def __init__(self, data: Data, **kwargs):
        kwargs.pop('collate_fn', None)
        batch_size = kwargs.get('batch_size', 1)

        output_nodes = self.get_output_nodes(self)

        if batch_size == 1:  # Pre-process subgraphs:
            data_list = ...
            super().__init__(data_list, collate_fn=self._cache_fn, **kwargs)
        else:
            self.data = data
            super().__init__(output_nodes, collate_fn=self._collate_fn,
                             **kwargs)

    def get_output_nodes(self) -> List[OutputNodes]:
        raise NotImplementedError

    def _cache_fn(self, data_list: List[Data]) -> Data:
        assert len(data_list) == 1
        return data_list[0]

    def _collate_fn(self, output_nodes: List[OutputNodes]) -> Data:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def get_partitions(
    edge_index: Union[Tensor, SparseTensor],
    num_partitions: int,
    indices: Tensor,
    num_nodes: int,
    output_weight: Optional[float] = None,
) -> List[Tensor]:
    assert isinstance(
        edge_index,
        (torch.LongTensor,
         SparseTensor)), f'Unsupported edge_index type {type(edge_index)}'
    if isinstance(edge_index, torch.LongTensor):
        edge_index = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(num_nodes, num_nodes))

    if output_weight is not None and output_weight != 1:
        node_weight = torch.ones(num_nodes)
        node_weight[indices] = output_weight
    else:
        node_weight = None

    _, partptr, perm = edge_index.partition(num_parts=num_partitions,
                                            recursive=False, weighted=False,
                                            node_weight=node_weight)

    partitions = []
    for i in range(len(partptr) - 1):
        partitions.append(perm[partptr[i]:partptr[i + 1]])

    return partitions


def get_pair_wise_distance(
    ys: List,
    num_classes: int,
    dist_type: str = 'kl',
) -> np.ndarray:
    num_batches = len(ys)

    counts = np.zeros((num_batches, num_classes), dtype=np.int32)
    for i in range(num_batches):
        unique, count = np.unique(ys[i], return_counts=True)
        counts[i, unique] = count

    counts += 1
    counts = counts / counts.sum(1).reshape(-1, 1)
    pairwise_dist = np.zeros((num_batches, num_batches), dtype=np.float64)

    for i in range(0, num_batches - 1):
        for j in range(i + 1, num_batches):
            if dist_type == 'l1':
                pairwise_dist[i, j] = np.sum(np.abs(counts[i] - counts[j]))
            elif dist_type == 'kl':

                def kl_divergence(p: np.ndarray, q: np.ndarray):
                    return (p * np.log(p / q)).sum()

                pairwise_dist[i, j] = kl_divergence(counts[i],
                                                    counts[j]) + kl_divergence(
                                                        counts[j], counts[i])
            else:
                raise ValueError

    pairwise_dist += pairwise_dist.T
    pairwise_dist += 1e-5  # for numerical stability
    np.fill_diagonal(pairwise_dist, 0.)

    return pairwise_dist


def indices_complete_check(
    loader: List[Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]],
    output_indices: Union[Tensor, np.ndarray],
):
    if isinstance(output_indices, Tensor):
        output_indices = output_indices.cpu().numpy()

    outs = []
    for out, aux in loader:
        if isinstance(out, Tensor):
            out = out.cpu().numpy()
        if isinstance(aux, Tensor):
            aux = aux.cpu().numpy()

        assert np.all(np.in1d(out,
                              aux)), "Not all output nodes are in aux nodes!"
        outs.append(out)

    outs = np.sort(np.concatenate(outs))
    assert np.all(
        outs == np.sort(output_indices)), "Output nodes missing or duplicate!"


def get_subgraph(
    out_indices: Tensor,
    graph: Data,
    return_edge_index_type: str,
    adj: SparseTensor,
    **kwargs,
):
    if return_edge_index_type == 'adj':
        assert adj is not None

    if return_edge_index_type == 'adj':
        subg = Data(x=graph.x[out_indices], y=graph.y[out_indices],
                    edge_index=adj[out_indices, :][:, out_indices])
    elif return_edge_index_type == 'edge_index':
        edge_index, edge_attr = subgraph(out_indices, graph.edge_index,
                                         graph.edge_attr, relabel_nodes=True,
                                         num_nodes=graph.num_nodes,
                                         return_edge_mask=False)
        subg = Data(x=graph.x[out_indices], y=graph.y[out_indices],
                    edge_index=edge_index, edge_attr=edge_attr)
    else:
        raise NotImplementedError

    for k, v in kwargs.items():
        subg[k] = v

    return subg


def define_sampler(
    batch_order: str,
    ys: List[Union[Tensor, np.ndarray, List]],
    num_classes: int,
    dist_type: str = 'kl',
):
    if batch_order == 'rand':
        logging.info("Running with random order")
        sampler = torch.utils.data.RandomSampler(ys)
    elif batch_order in ['order', 'sample']:
        kl_div = get_pair_wise_distance(ys, num_classes, dist_type=dist_type)
        if batch_order == 'order':
            from python_tsp.heuristics import solve_tsp_simulated_annealing
            best_perm, _ = solve_tsp_simulated_annealing(kl_div)
            logging.info(f"Running with given order: {best_perm}")
            sampler = IBMBOrderedSampler(best_perm)
        else:
            logging.info("Running with weighted sampling")
            sampler = IBMBWeightedSampler(kl_div)
    else:
        raise ValueError

    return sampler


def create_batchwise_out_aux_pairs(
    adj: SparseTensor,
    partitions: List[Union[torch.LongTensor, np.ndarray]],
    prime_indices: Union[torch.LongTensor, np.ndarray],
    topk: int,
    num_outnodeset_per_batch: int = 50,
    alpha: float = 0.2,
    ppr_iterations: int = 50,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    def ppr_power_method(
        adj: SparseTensor,
        batch: List[Union[np.ndarray, torch.LongTensor]],
        topk: int,
        num_iter: int,
        alpha: float,
    ) -> List[np.ndarray]:

        topk_neighbors = []
        logits = torch.zeros(
            adj.size(0), len(batch),
            device=adj.device())  # each column contains a set of output nodes
        for i, tele_set in enumerate(batch):
            logits[tele_set, i] = 1. / len(tele_set)

        new_logits = logits.clone()
        for i in range(num_iter):
            new_logits = adj @ new_logits * (1 - alpha) + alpha * logits

        inds = new_logits.argsort(0)
        nonzeros = (new_logits > 0).sum(0)
        nonzeros = torch.minimum(
            nonzeros,
            torch.tensor([topk], dtype=torch.int64, device=adj.device()))
        for i in range(new_logits.shape[1]):
            topk_neighbors.append(inds[-nonzeros[i]:, i].cpu().numpy())

        return topk_neighbors

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(prime_indices, Tensor):
        prime_indices = prime_indices.cpu().numpy()

    adj = adj.to(device)

    cur_output_nodes = []
    loader = []

    pbar = tqdm(range(len(partitions)))
    pbar.set_description("Processing topic-sensitive PPR batches")
    for n in pbar:
        part = partitions[n]
        if isinstance(part, Tensor):
            part = part.cpu().numpy()

        primes_in_part, *_ = np.intersect1d(part, prime_indices,
                                            assume_unique=True,
                                            return_indices=True)
        if len(primes_in_part):  # no output nodes in this partition
            cur_output_nodes.append(primes_in_part)

        # accumulate enough output nodes to make good use of GPU memory
        if len(cur_output_nodes
               ) >= num_outnodeset_per_batch or n == len(partitions) - 1:
            topk_neighbors = ppr_power_method(adj, cur_output_nodes, topk,
                                              ppr_iterations, alpha)
            for i in range(len(cur_output_nodes)):
                # force output nodes to be aux nodes
                auxiliary_nodes = np.union1d(cur_output_nodes[i],
                                             topk_neighbors[i])
                loader.append((cur_output_nodes[i], auxiliary_nodes))
            cur_output_nodes = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return loader


def get_pairs(ppr_mat: scipy.sparse.csr_matrix) -> np.ndarray:
    ppr_mat = ppr_mat + ppr_mat.transpose()

    ppr_mat = ppr_mat.tocoo()
    row, col, data = ppr_mat.row, ppr_mat.col, ppr_mat.data
    mask = (row > col)  # lu

    row, col, data = row[mask], col[mask], data[mask]
    sort_arg = np.argsort(data)[::-1]
    # sort_arg = parallel_sort.parallel_argsort(data)[::-1]

    # map prime_nodes to arange
    ppr_pairs = np.vstack((row[sort_arg], col[sort_arg])).T
    return ppr_pairs


_prime_orient_merge_numba: Optional[Callable] = None


def prime_orient_merge(
    ppr_pairs: np.ndarray,
    primes_per_batch: int,
    num_nodes: int,
):
    if not WITH_NUMBA:  # pragma: no cover
        raise ImportError("'prime_orient_merge' requires the 'numba' package")

    global _prime_orient_merge_numba
    if _prime_orient_merge_numba is None:
        _prime_orient_merge_numba = numba.njit(cache=True)(_prime_orient_merge)

    return _prime_orient_merge_numba(ppr_pairs, primes_per_batch, num_nodes)


def _prime_orient_merge(
    ppr_pairs: np.ndarray,
    primes_per_batch: int,
    num_nodes: int,
):
    id_primes_list = list(np.arange(num_nodes, dtype=np.int32).reshape(-1, 1))
    node_id_list = np.arange(num_nodes, dtype=np.int32)
    placeholder = np.zeros(0, dtype=np.int32)

    for i, j in ppr_pairs:
        id1, id2 = node_id_list[i], node_id_list[j]
        if id1 > id2:
            id1, id2 = id2, id1

        if id1 != id2 and len(id_primes_list[id1]) + len(
                id_primes_list[id2]) <= primes_per_batch:
            id_primes_list[id1] = np.concatenate(
                (id_primes_list[id1], id_primes_list[id2]))
            node_id_list[id_primes_list[id2]] = id1
            id_primes_list[id2] = placeholder

    prime_lst = list()
    ids = np.unique(node_id_list)

    for _id in ids:
        prime_lst.append(list(id_primes_list[_id]))

    return list(prime_lst)


def prime_post_process(loader, merge_max_size):
    from heapq import heapify, heappop, heappush

    h = [(
        len(p),
        p,
    ) for p in loader]
    heapify(h)

    while len(h) > 1:
        len1, p1 = heappop(h)
        len2, p2 = heappop(h)
        if len1 + len2 <= merge_max_size:
            heappush(h, (len1 + len2, p1 + p2))
        else:
            heappush(h, (
                len1,
                p1,
            ))
            heappush(h, (
                len2,
                p2,
            ))
            break

    new_batch = []

    while len(h):
        _, p = heappop(h)
        new_batch.append(p)

    return new_batch


def topk_ppr_matrix(
    edge_index: Tensor,
    num_nodes: int,
    alpha: float,
    eps: float,
    output_node_indices: Union[np.ndarray, torch.LongTensor],
    topk: int,
    normalization='row',
) -> Tuple[scipy.sparse.csr_matrix, List[np.ndarray]]:
    neighbors, weights = get_ppr(edge_index, alpha, eps, output_node_indices,
                                 num_nodes)

    _, neighbor_counts = neighbors[0].unique(return_counts=True)

    ppr_matrix = SparseTensor(
        row=torch.arange(
            len(output_node_indices)).repeat_interleave(neighbor_counts),
        col=neighbors[1], value=weights,
        sparse_sizes=(len(output_node_indices),
                      num_nodes)).to_scipy(layout='csr')

    neighbors = [
        n.cpu().numpy()
        for n in torch.split(neighbors[1],
                             neighbor_counts.cpu().tolist(), dim=0)
    ]
    weights = [
        n.cpu().numpy()
        for n in torch.split(weights,
                             neighbor_counts.cpu().tolist(), dim=0)
    ]

    def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray],
                 topk: int):
        new_neighbors = []
        for n, w in zip(neighbors, weights):
            idx_topk = np.argsort(w)[-topk:]
            new_neighbor = n[idx_topk]
            new_neighbors.append(new_neighbor)

        return new_neighbors

    neighbors = sparsify(neighbors, weights, topk)
    neighbors = [
        np.union1d(nei, pr) for nei, pr in zip(neighbors, output_node_indices)
    ]

    _, out_degree = torch.unique(edge_index[0], sorted=True,
                                 return_counts=True)
    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(out_degree, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = deg_sqrt[output_node_indices[row]] * \
            ppr_matrix.data * \
            deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(out_degree, 1e-12)

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = out_degree[output_node_indices[row]] * \
            ppr_matrix.data * \
            deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return ppr_matrix, neighbors


class IBMBBaseLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data_list: Union[List[Data], List[Tuple]],
        graph: Data,
        adj: SparseTensor,
        return_edge_index_type: str,
        **kwargs,
    ):
        self.graph = graph
        self.adj = adj
        self.return_edge_index_type = return_edge_index_type
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        super().__init__(data_list, collate_fn=self.collate_fn, **kwargs)

    def create_loader(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def prepare_cache(
        cls,
        graph: Data,
        batch_wise_out_aux_pairs: List[Tuple[np.ndarray, np.ndarray]],
        adj: Optional[SparseTensor],
        return_edge_index_type: str,
    ):
        subgraphs = []

        pbar = tqdm(batch_wise_out_aux_pairs)
        pbar.set_description(
            f"Caching data with type {return_edge_index_type}")

        if return_edge_index_type == 'adj':
            assert adj is not None

        for out, aux in pbar:
            mask = torch.from_numpy(np.in1d(aux, out))
            if isinstance(aux, np.ndarray):
                aux = torch.from_numpy(aux)
            subg = get_subgraph(aux, graph, return_edge_index_type, adj,
                                output_node_mask=mask)
            subgraphs.append(subg)

        return subgraphs

    @classmethod
    def create_adj_from_edge_index(
        cls,
        edge_index: Tensor,
        num_nodes: int,
        normalization: str,
    ):
        assert normalization in ['sym', 'rw']
        adj = SparseTensor.from_edge_index(
            edge_index,
            sparse_sizes=(num_nodes, num_nodes),
        )
        adj = adj.fill_value(1.)
        degree = adj.sum(0)

        degree[degree == 0.] = 1e-12
        deg_inv = 1 / degree

        if normalization == 'sym':
            deg_inv_sqrt = deg_inv**0.5
            adj = adj * deg_inv_sqrt.reshape(1, -1)
            adj = adj * deg_inv_sqrt.reshape(-1, 1)
        elif normalization == 'rw':
            adj = adj * deg_inv.reshape(-1, 1)

        return adj

    def collate_fn(self, data_list: List[Union[Data, Tuple]]):
        if len(data_list) == 1 and isinstance(data_list[0], Data):
            return data_list[0]

        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = np.unique(np.concatenate(aux))
        mask = torch.from_numpy(np.in1d(aux, out))
        aux = torch.from_numpy(aux)

        subg = get_subgraph(aux, self.graph, self.return_edge_index_type,
                            self.adj, output_node_mask=mask)
        return subg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class IBMBBatchLoader(IBMBBaseLoader):
    r"""The batch-wise influence-based data loader from the
    `"Influence-Based Mini-Batching for Graph Neural Networks"
    <https://arxiv.org/abs/2212.09083>`__ paper.

    First, the METIS graph partitioning algorithm separates the graph into
    :obj:`num_partitions` many partitions.
    Afterwards, input/seed nodes and their auxiliary nodes (found via
    topic-sensitive PageRank) are used to form a mini-batch.

    If :obj:`batch_size` is set to :obj:`1`, mini-batches are pre-calculated
    and cached in memory.
    Otherwise, only input nodes and their auxiliary nodes are pre-computed, and
    mini-batches are collated on-the-fly.

    Args:
        data (torch_geometric.data.Data): A
            :class:`~torch_geometric.data.Data` object.
        batch_order (str): A string indicating the batch order type (one of
            :obj:`"order"`, :obj:`"sample"` or :obj:`"rand"`).
            If :obj:`"order"`, calculates the pair-wise KL divergence between
            every two batches to organize an optimal order.
            If :obj:`"sample"`, samples the next batch w.r.t. the last one in
            which a batch with higher KL divergence score is more likely to be
            sampled.
            If :obj:`"rand"`, batches are generated randomly.
        num_partitions (int): The number of partitions.
        input_nodes (torch.Tensor): A vector containing the set of seed
            nodes.
        batch_expand_ratio (float, optional): The ratio between the returned
            batch size and the original partition size. For example, set it to
            :obj:`2.0` in case you would like the batch to have double the
            number of nodes as the size of its partition.
            (default: :obj:`1.0`)
        metis_input_node_weight (float, optional): The weights on the input
            nodes for METIS graph partitioning. (default: :obj:`None`)
        alpha (float, optional): The teleport probability of the PageRank
            calculation. (default: :obj:`0.2`)
        approximate_ppr_iterations (int, optional): The number of power
            iterations for PageRank calculation. (default: :obj:`50`)
        return_edge_index_type (str, optional): A string indicating the output
            type of edge indices (one of :obj:`"edge_index"` or :obj:`"adj"`).
            If set to :obj:`"adj"`, the :obj:`edge_index` of the batch will
            be a :class:`torch_sparse.SparseTensor`, otherwise a
            :class:`torch.Tensor`. (default: :obj:`"edge_index"`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Data,
        batch_order: str,
        num_partitions: int,
        input_nodes: Tensor,
        batch_expand_ratio: Optional[float] = 1.0,
        metis_input_node_weight: Optional[float] = None,
        alpha: Optional[float] = 0.2,
        approximate_ppr_iterations: Optional[int] = 50,
        return_edge_index_type: str = 'edge_index',
        **kwargs,
    ):
        self.subgraphs = []
        self.batch_wise_out_aux_pairs = []

        assert is_undirected(
            data.edge_index,
            num_nodes=data.num_nodes), "Assume the graph to be undirected"
        assert batch_order in ['rand', 'sample', 'order'
                               ], f"Unsupported batch order: {batch_order}"

        adj = self.create_adj_from_edge_index(
            data.edge_index,
            data.num_nodes,
            normalization='rw',
        )

        self.cache_data = kwargs['batch_size'] == 1
        self.num_partitions = num_partitions
        self.output_indices = input_nodes
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.batch_expand_ratio = batch_expand_ratio
        self.metis_output_weight = metis_input_node_weight
        self.num_outnodeset_per_batch = 50
        self.alpha = alpha
        self.approximate_ppr_iterations = approximate_ppr_iterations

        self.create_loader(data, adj)

        if len(self.batch_wise_out_aux_pairs) > 2:  # <= 2 order makes no sense
            ys = [
                data.y[out].numpy() for out, _ in self.batch_wise_out_aux_pairs
            ]
            sampler = define_sampler(batch_order, ys, data.y.max().item() + 1)
        else:
            sampler = None

        if not self.cache_data:
            cached_data = data  # need to cache the original graph
            if return_edge_index_type == 'adj':
                cached_adj = adj
            else:
                cached_adj = None
        else:
            cached_data = None
            cached_adj = None

        super().__init__(
            self.subgraphs
            if self.cache_data else self.batch_wise_out_aux_pairs,
            cached_data,
            cached_adj,
            return_edge_index_type,
            sampler=sampler,
            **kwargs,
        )

    def create_loader(self, graph: Data, adj: SparseTensor):
        partitions = get_partitions(
            adj,
            self.num_partitions,
            self.output_indices,
            graph.num_nodes,
            self.metis_output_weight,
        )

        # get output - auxiliary node pairs
        topk = math.ceil(self.batch_expand_ratio * graph.num_nodes /
                         self.num_partitions)
        batch_wise_out_aux_pairs = create_batchwise_out_aux_pairs(
            adj, partitions, self.output_indices, topk,
            self.num_outnodeset_per_batch, self.alpha,
            self.approximate_ppr_iterations)

        indices_complete_check(batch_wise_out_aux_pairs, self.output_indices)
        self.batch_wise_out_aux_pairs = batch_wise_out_aux_pairs

        if self.cache_data:
            self.subgraphs = self.prepare_cache(
                graph,
                batch_wise_out_aux_pairs,
                adj,
                self.return_edge_index_type,
            )


class IBMBNodeLoader(IBMBBaseLoader):
    r"""The node-wise influence-based data loader from the
    `"Influence-Based Mini-Batching for Graph Neural Networks"
    <https://arxiv.org/abs/2212.09083>`__ paper.

    First, the Personalized PageRank (PPR) score for each input node is
    computed, for which the :obj:`k` nodes with the highest scores are taken
    auxiliary nodes.
    Afterwards, input nodes are merged according to their pair-wise PPR scores.

    Similar to :class:`~torch_geometric.loader.IBMBBatchLoader`, subgraphs are
    cached in memory for :obj:`batch_size = 1`, and collated on-the-fly
    otherwise.

    Args:
        data (torch_geometric.data.Data): A
            :class:`~torch_geometric.data.Data` object.
        batch_order (str): A string indicating the batch order type (one of
            :obj:`"order"`, :obj:`"sample"` or :obj:`"rand"`).
            If :obj:`"order"`, calculates the pair-wise KL divergence between
            every two batches to organize an optimal order.
            If :obj:`"sample"`, samples the next batch w.r.t. the last one in
            which a batch with higher KL divergence score is more likely to be
            sampled.
            If :obj:`"rand"`, batches are generated randomly.
        input_nodes (torch.Tensor): A vector containing the set of seed
            nodes.
        num_auxiliary_nodes (int): The number of auxiliary nodes per input
            node.
        num_nodes_per_batch (int): The number of seed nodes per batch.
        alpha (float, optional): The teleport probability of the PageRank
            calculation. (default: :obj:`0.2`)
        eps (float, optional): The threshold for stopping the PPR calculation
            The smaller :obj`eps` is, the more accurate are the results of
            PPR calculation, but it also takes longer.
            (default: :obj:`1e-5`)
        return_edge_index_type (str, optional): A string indicating the output
            type of edge indices (one of :obj:`"edge_index"` or :obj:`"adj"`).
            If set to :obj:`"adj"`, the :obj:`edge_index` of the batch will
            be a :class:`torch_sparse.SparseTensor`, otherwise a
            :class:`torch.Tensor`. (default: :obj:`"edge_index"`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Data,
        batch_order: str,
        input_nodes: torch.Tensor,
        num_auxiliary_nodes: int,
        num_nodes_per_batch: int,
        alpha: float = 0.2,
        eps: float = 1e-5,
        return_edge_index_type: str = 'edge_index',
        **kwargs,
    ):
        self.subgraphs = []
        self.node_wise_out_aux_pairs = []

        assert is_undirected(
            data.edge_index,
            num_nodes=data.num_nodes), "Assume the graph to be undirected"
        assert batch_order in ['rand', 'sample', 'order'
                               ], f"Unsupported batch order: {batch_order}"

        if return_edge_index_type == 'adj':
            adj = self.create_adj_from_edge_index(data.edge_index,
                                                  data.num_nodes,
                                                  normalization='rw')
        else:
            adj = None

        self.cache_data = kwargs['batch_size'] == 1
        self._batchsize = kwargs['batch_size']
        self.output_indices = input_nodes.numpy()
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.num_auxiliary_node_per_output = num_auxiliary_nodes
        self.num_output_nodes_per_batch = num_nodes_per_batch
        self.alpha = alpha
        self.eps = eps

        self.create_loader(data, adj)

        if len(self.node_wise_out_aux_pairs) > 2:  # <= 2 order makes no sense
            ys = [
                data.y[out].numpy() for out, _ in self.node_wise_out_aux_pairs
            ]
            sampler = define_sampler(batch_order, ys, data.y.max().item() + 1)
        else:
            sampler = None

        if not self.cache_data:
            cached_graph = data  # need to cache the original graph
            cached_adj = adj
        else:
            cached_graph = None
            cached_adj = None

        super().__init__(
            self.subgraphs
            if self.cache_data else self.node_wise_out_aux_pairs,
            cached_graph,
            cached_adj,
            return_edge_index_type,
            sampler=sampler,
            **kwargs,
        )

    def create_loader(self, graph: Data, adj: SparseTensor):
        logging.info("Start PPR calculation")
        ppr_matrix, neighbors = topk_ppr_matrix(
            graph.edge_index, graph.num_nodes, self.alpha, self.eps,
            torch.from_numpy(self.output_indices),
            self.num_auxiliary_node_per_output)

        ppr_matrix = ppr_matrix[:, self.output_indices]

        logging.info("Getting PPR pairs")
        ppr_pairs = get_pairs(ppr_matrix)

        output_list = prime_orient_merge(
            ppr_pairs,
            self.num_output_nodes_per_batch,
            len(self.output_indices),
        )
        output_list = prime_post_process(
            output_list,
            self.num_output_nodes_per_batch,
        )
        node_wise_out_aux_pairs = []

        if isinstance(neighbors, list):
            neighbors = np.array(neighbors, dtype=object)

        def _union(inputs):
            return np.unique(np.concatenate(inputs))

        for p in output_list:
            node_wise_out_aux_pairs.append(
                (self.output_indices[p],
                 _union(neighbors[p]).astype(np.int64)))

        indices_complete_check(node_wise_out_aux_pairs, self.output_indices)
        self.node_wise_out_aux_pairs = node_wise_out_aux_pairs

        if self.cache_data:
            self.subgraphs = self.prepare_cache(
                graph,
                node_wise_out_aux_pairs,
                adj,
                self.return_edge_index_type,
            )


class IBMBOrderedSampler(torch.utils.data.Sampler[int]):
    r"""A sampler with given order, specially for IBMB loaders.

    Args:
        data_source (np.ndarray, torch.Tensor, List): A :obj:`np.ndarray`,
            :obj:`torch.Tensor`, or :obj:`List` data object. Contains the
            order of the batches.
    """
    def __init__(self, data_source: Union[np.ndarray, torch.Tensor,
                                          List]) -> None:
        self.data_source = data_source
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)


class IBMBWeightedSampler(torch.utils.data.Sampler[int]):
    r"""A weighted sampler wrt the pair wise KL divergence.
    The very first batch after initialization is sampled randomly,
    with the next ones being sampled according to the last batch,
    including the first batch in the next round.

    Args:
        batch_kl_div (np.ndarray, torch.Tensor): A :obj:`np.ndarray` or
            :obj:`torch.Tensor`, each element [i, j] contains the pair wise
            KL divergence between batch i and j.
    """
    def __init__(self, batch_kl_div: Union[np.ndarray, torch.Tensor]) -> None:
        data_source = np.arange(batch_kl_div.shape[0])
        self.data_source = data_source
        self.batch_kl_div = batch_kl_div
        self.last_train_batch_id = 0
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        probs = self.batch_kl_div.copy()

        last = self.last_train_batch_id
        num_batches = probs.shape[0]

        fetch_idx = []

        next_id = 0
        while np.any(probs):
            next_id = np.random.choice(num_batches, size=None, replace=False,
                                       p=probs[last] / probs[last].sum())
            last = next_id
            fetch_idx.append(next_id)
            probs[:, next_id] = 0.

        self.last_train_batch_id = next_id

        return iter(fetch_idx)

    def __len__(self) -> int:
        return len(self.data_source)
