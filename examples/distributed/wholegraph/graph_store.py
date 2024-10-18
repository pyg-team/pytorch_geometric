from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.distributed as dist
from nv_distributed_graph import DistGraphCSC, dist_shmem, nvlink_network

import torch_geometric
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.typing import EdgeType


@dataclass
class WholeGraphEdgeAttr(EdgeAttr):
    r"""Edge attribute class for WholeGraph GraphStore enforcing layout to be CSC."""
    def __init__(
        self,
        edge_type: Optional[EdgeType] = None,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        layout = EdgeLayout.CSC  # Enforce CSC layout for WholeGraph for now
        super().__init__(edge_type, layout, is_sorted, size)


class WholeGraphGraphStore(GraphStore):
    r"""A high-performance, UVA-enabled, and multi-GPU/multi-node friendly graph store, powered by WholeGraph library.
    It is compatible with PyG's GraphStore base class and supports both homogeneous and heterogeneous graph data types.

    Args:
        pyg_data (torch_geometric.data.Data or torch_geometric.data.HeteroData): The input PyG graph data.
        format (str): The underlying graph format to use. Default is 'wholegraph'.


    """
    def __init__(self, pyg_data, format='wholegraph'):
        super().__init__(edge_attr_cls=WholeGraphEdgeAttr)
        self._g = {
        }  # for simplicy, _g is a dictionary of DistGraphCSC to hold the graph structure data for each type

        if format == 'wholegraph':
            pinned_shared = False
            if dist_shmem.get_local_size() == dist.get_world_size():
                backend = 'vmm'
            else:
                backend = 'vmm' if nvlink_network() else 'nccl'
        elif format == 'pyg':
            pinned_shared = True
            backend = None  # backend is a no-op for pyg format
        else:
            raise ValueError("Unsupported underlying graph format")

        if isinstance(pyg_data, torch_geometric.data.Data):
            # issue: this will crash: pyg_data.get_all_edge_attrs()[0] if pyg_data is a torch sparse csr
            # walkaround:
            if 'adj_t' not in pyg_data:
                row, col = None, None
                if dist_shmem.get_local_rank() == 0:
                    row, col, _ = pyg_data.csc()
                row = dist_shmem.to_shmem(row)
                col = dist_shmem.to_shmem(col)
                size = pyg_data.size()
            else:
                col = pyg_data.adj_t.crow_indices()
                row = pyg_data.adj_t.col_indices()
                size = pyg_data.adj_t.size()[::-1]

            self.num_nodes = pyg_data.num_nodes
            graph = DistGraphCSC(
                col,
                row,
                device="cpu",
                backend=backend,
                pinned_shared=pinned_shared,
            )
            self.put_adj_t(graph, size=size)

        elif isinstance(pyg_data,
                        torch_geometric.data.HeteroData):  # hetero graph
            # issue: this will crash: pyg_data.get_all_edge_attrs()[0] if pyg_data is a torch sparse csr
            # walkaround:
            self.num_nodes = pyg_data.num_nodes
            for edge_type, edge_store in pyg_data.edge_items():
                if 'adj_t' not in edge_store:
                    row, col = None, None
                    if dist_shmem.get_local_rank() == 0:
                        row, col, _ = edge_store.csc()
                    row = dist_shmem.to_shmem(row)
                    col = dist_shmem.to_shmem(col)
                    size = edge_store.size()
                else:
                    col = edge_store.adj_t.crow_indices()
                    row = edge_store.adj_t.col_indices()
                    size = edge_store.adj_t.size()[::-1]
                    graph = DistGraphCSC(
                        col,
                        row,
                        device="cpu",
                        backend=backend,
                        pinned_shared=pinned_shared,
                    )
                    self.put_adj_t(graph, edge_type=edge_type, size=size)

    def put_adj_t(self, adj_t: DistGraphCSC, *args, **kwargs) -> bool:
        """Add an adj_t (adj with transpose) matrix, :obj:`DistGraphCSC`
        to :class:`WholeGraphGraphStore`.
        Returns whether insertion was successful.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._put_adj_t(adj_t, edge_attr)

    def get_adj_t(self, *args, **kwargs) -> DistGraphCSC:
        """Retrieves an adj_t (adj with transpose) matrix, :obj:`DistGraphCSC`
        from :class:`WholeGraphGraphStore`.
        Return: :obj:`DistGraphCSC`
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        graph_adj_t = self._get_adj_t(edge_attr)
        if graph_adj_t is None:
            raise KeyError(f"'adj_t' for '{edge_attr}' not found")
        return graph_adj_t

    def _put_adj_t(self, adj_t: DistGraphCSC,
                   edge_attr: WholeGraphEdgeAttr) -> bool:
        if not hasattr(self, '_edge_attrs'):
            self._edge_attrs = {}
        self._edge_attrs[edge_attr.edge_type] = edge_attr

        self._g[edge_attr.edge_type] = adj_t
        if edge_attr.size is None:
            # Hopefully, the size is already set beforehand by the input, edge_attr
            # Todo: DistGraphCSC does not support size attribute, need to implement it
            edge_attr.size = adj_t.size
        return True

    def _get_adj_t(self,
                   edge_attr: WholeGraphEdgeAttr) -> Optional[DistGraphCSC]:
        store = self._g.get(edge_attr.edge_type)
        edge_attrs = getattr(self, '_edge_attrs', {})
        edge_attr = edge_attrs[edge_attr.edge_type]
        if edge_attr.size is None:
            # Hopefully, the size is already set beforehand by the input, edge_attr
            # Todo: DistGraphCSC does not support size attribute, need to implement it
            edge_attr.size = store.size  # Modify in-place.
        return store

    def _get_edge_index():
        pass

    def _put_edge_index():
        pass

    def _remove_edge_index():
        pass

    def __getitem__(self, key: WholeGraphEdgeAttr):
        return self.get_adj_t(key)

    def __setitem__(self, key: WholeGraphEdgeAttr, value: DistGraphCSC):
        self.put_adj_t(value, key)

    def get_all_edge_attrs(self) -> List[WholeGraphEdgeAttr]:
        edge_attrs = getattr(self, '_edge_attrs', {})
        for key, store in self._g.items():
            if key not in edge_attrs:
                edge_attrs[key] = WholeGraphEdgeAttr(key, size=store.size)
        return list(edge_attrs.values())

    def csc(self):
        # Define this method to be compatible with pyg native neighbor sampler (if used) see: sampler/neighbour_sampler.py:L222 and L263
        if not self.is_hetero:
            key = self.get_all_edge_attrs()[0]
            store = self._get_adj_t(key)
            return store.row_indx, store.col_ptrs, None  # no permutation vector
        else:
            row_dict = {}
            col_dict = {}
            for edge_attr in self.get_all_edge_attrs():
                store = self._get_adj_t(edge_attr)
                row_dict[edge_attr.edge_type] = store.row_indx
                col_dict[edge_attr.edge_type] = store.col_ptrs
            return row_dict, col_dict, None

    @property
    def is_hetero(self):
        if len(self._g) > 1:
            return True
        return False

    @staticmethod
    def create_pyg_subgraph(WG_SampleOutput) -> Tuple:
        # PyG_SampleOutput (node, row, col, edge, batch...):
        # node (torch.Tensor): The sampled nodes in the original graph.
        # row (torch.Tensor): The source node indices of the sampled subgraph.
        #                     Indices must be within {0, ..., num_nodes - 1} where num_nodes is the number of nodes in sampled graph.
        # col (torch.Tensor): The destination node indices of the sampled subgraph. Indices must be within {0, ..., num_nodes - 1}
        # edge (torch.Tensor, optional): The sampled edges in the original graph. (for obtain edge features from the original graph)
        # batch (torch.Tensor, optional): The vector to identify the seed node for each sampled node in case of disjoint subgraph
        #                                  sampling per seed node. (None)
        # num_sampled_nodes (List[int], optional): The number of sampled nodes per hop.
        # num_sampled_edges (List[int], optional): The number of sampled edges per hop.
        sampled_nodes_list, edge_indice_list, csr_row_ptr_list, csr_col_ind_list = WG_SampleOutput
        num_sampled_nodes = []
        node = sampled_nodes_list[0]

        for hop in range(len(sampled_nodes_list) - 1):
            sampled_nodes = len(sampled_nodes_list[hop]) - len(
                sampled_nodes_list[hop + 1])
            num_sampled_nodes.append(sampled_nodes)
        num_sampled_nodes.append(len(sampled_nodes_list[-1]))
        num_sampled_nodes.reverse()

        layers = len(edge_indice_list)
        num_sampled_edges = [len(csr_col_ind_list[-1])]
        # Loop in reverse order, starting from the second last layer
        for layer in range(layers - 2, -1, -1):
            num_sampled_edges.append(
                len(csr_col_ind_list[layer] -
                    len(csr_col_ind_list[layer + 1])))

        row = csr_col_ind_list[0]  # rows
        col = edge_indice_list[0][1]  # dst node

        edge = None
        batch = None
        out = node, row, col, edge, batch, num_sampled_nodes, num_sampled_edges
        return SamplerOutput.cast(out)
