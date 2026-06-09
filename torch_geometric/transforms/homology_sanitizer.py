from typing import Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('homology_sanitizer')
class HomologySanitizer(BaseTransform):
    r"""Detects and removes anomalously dense cyclic subgraphs (topological
    homology backdoor triggers) from a homogeneous or heterogeneous graph.

    This transform uses a two-phase cascade:

    1. **Community Anomaly Screening:** Computes the normalized cycle density
       of each connected component. Components with density exceeding
       :obj:`community_cycle_ratio` are flagged as suspicious.
    2. **Ego-Network Surgical Audit:** Inside suspicious components, extracts
       the :obj:`ego_hops`-hop ego network for each node and computes its
       exact Betti-1 number (independent cycle count). Nodes whose ego
       network has a Betti-1 number ≥ :obj:`threshold` are flagged as
       anomalous, and all edges between pairs of anomalous nodes are
       removed.

    For massive graphs (:obj:`num_nodes > fast_path_limit`), Phase 2 is
    skipped and only Phase 1 is used. Setting :obj:`use_exact=False`
    also disables Phase 2 unconditionally.

    Args:
        threshold (int): Minimum Betti-1 number of an ego network to flag
            its center node as anomalous. (default: :obj:`2`)
        ego_hops (int): Hop radius for ego-network extraction in Phase 2.
            (default: :obj:`2`)
        community_cycle_ratio (float): Normalized cycle-density threshold
            for Phase 1. A component is suspicious if
            :obj:`(E - V + C) / V > community_cycle_ratio`.
            (default: :obj:`0.3`)
        fast_path_limit (int, optional): If :obj:`data.num_nodes` exceeds
            this value, skip Phase 2. :obj:`None` disables the fast path.
            (default: :obj:`5000`)
        use_exact (bool): Whether to run Phase 2 at all. If :obj:`False`,
            only Phase 1 is used regardless of graph size.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        threshold: int = 2,
        ego_hops: int = 2,
        community_cycle_ratio: float = 0.3,
        fast_path_limit: Optional[int] = 5000,
        use_exact: bool = True,
    ) -> None:
        self.threshold = threshold
        self.ego_hops = ego_hops
        self.community_cycle_ratio = community_cycle_ratio
        self.fast_path_limit = fast_path_limit
        self.use_exact = use_exact

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if 'edge_index' not in store or store.edge_index.numel() == 0:
                continue

            edge_index = store.edge_index

            # Determine node-space size for this edge store
            size = [s for s in store.size() if s is not None]
            num_nodes = max(size) if len(size) > 0 else None
            if num_nodes is None or num_nodes == 0:
                continue

            # Convert to undirected for cycle analysis
            edge_index_undirected = to_undirected(edge_index,
                                                  num_nodes=num_nodes)
            edge_list = edge_index_undirected.t().cpu().numpy()

            # Import optional dependencies lazily
            try:
                import networkx as nx
                import numpy as np
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
            except ImportError as e:
                raise ImportError(
                    f"'{self.__class__.__name__}' requires 'networkx' "
                    f"and 'scipy'. Install them via 'pip install "
                    f"networkx scipy'.") from e

            # --- Phase 1: Community Anomaly Screening ---
            adj = csr_matrix(
                (np.ones(edge_list.shape[0]),
                 (edge_list[:, 0], edge_list[:, 1])),
                shape=(num_nodes, num_nodes),
            )
            # Symmetrise (undirected) and binarise
            adj = adj.maximum(adj.T)
            adj.data = np.ones_like(adj.data)

            n_components, labels = connected_components(
                csgraph=adj, directed=False, return_labels=True)

            # Compute cycle density per component and collect suspicious nodes
            suspicious_nodes = set()
            for comp_id in range(n_components):
                nodes_in_comp = np.where(labels == comp_id)[0]
                v = int(nodes_in_comp.shape[0])
                if v == 0:
                    continue

                # Count unique undirected edges internal to the component
                mask_src = np.isin(edge_list[:, 0], nodes_in_comp)
                mask_dst = np.isin(edge_list[:, 1], nodes_in_comp)
                internal_edges = edge_list[mask_src & mask_dst]
                e = int(internal_edges.shape[0]) // 2

                cycle_density = (e - v + 1) / v
                if cycle_density > self.community_cycle_ratio:
                    suspicious_nodes.update(nodes_in_comp.tolist())

            # --- Phase 2: Ego-Network Surgical Audit ---
            anomalous_nodes = set()
            run_exact = (self.use_exact
                         and (self.fast_path_limit is None
                              or num_nodes <= self.fast_path_limit)
                         and len(suspicious_nodes) > 0)

            if run_exact:
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                G.add_edges_from(edge_list.tolist())

                for node in suspicious_nodes:
                    ego = nx.ego_graph(G, node, radius=self.ego_hops)
                    v_ego = ego.number_of_nodes()
                    if v_ego == 0:
                        continue

                    # Quick lower-bound: a tree has E = V - 1
                    e_ego = ego.number_of_edges()
                    if e_ego < v_ego + self.threshold - 1:
                        continue

                    c_ego = nx.number_connected_components(ego)
                    betti_1 = e_ego - v_ego + c_ego
                    if betti_1 >= self.threshold:
                        anomalous_nodes.add(node)
            elif not self.use_exact and len(suspicious_nodes) > 0:
                # Fallback: treat all suspicious nodes as anomalous
                anomalous_nodes = suspicious_nodes

            # --- Mitigation: prune edges between anomalous nodes ---
            if len(anomalous_nodes) == 0:
                continue

            anomalous_tensor = torch.tensor(
                sorted(anomalous_nodes),
                dtype=torch.long,
                device=edge_index.device,
            )
            is_anomalous = torch.zeros(
                num_nodes,
                dtype=torch.bool,
                device=edge_index.device,
            )
            is_anomalous[anomalous_tensor] = True

            # Build mask on the *original* edge_index (preserves direction)
            src = edge_index[0]
            dst = edge_index[1]
            keep_mask = ~(is_anomalous[src] & is_anomalous[dst])

            # Collect edge attributes *before* mutating edge_index so that
            # is_edge_attr() still evaluates against the original num_edges.
            edge_attr_keys = [
                key for key in store.keys()
                if key != 'edge_index' and store.is_edge_attr(key)
            ]

            store.edge_index = edge_index[:, keep_mask]

            # Filter edge-level attributes in lock-step
            for key in edge_attr_keys:
                store[key] = store[key][keep_mask]

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'threshold={self.threshold}, '
                f'ego_hops={self.ego_hops}, '
                f'community_cycle_ratio={self.community_cycle_ratio})')
