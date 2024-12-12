import logging
from collections import defaultdict

import numpy as np
import torch
from pyg_graphons import Graphon

from torch_geometric.data import Data, InMemoryDataset


class MixupDataset(InMemoryDataset):
    def __init__(self, dataset: InMemoryDataset,
                 specs: list[tuple[int, int, float, int, int, int,
                                   int]] = None,
                 estimation_method: str = 'usvt', **args):
        r"""The Mixup Dataset method from the `G-Mixup: Graph Data Augmentation for Graph Classification.
        <https://proceedings.mlr.press/v162/han22c.html>`_ paper.

        G-Mixup estimates Graphons (as defined in the paper) for every class in a dataset. It then linearly interpolates in the graphon space, and samples new points from intrpolated graphons to achieve Mixup-like linear interpolation despite an inherently non-linear data type (graphs).

        Concretely, this works in four steps:

        1. Graphon Estimation: estimate W_k such that G_k ∼ W_k is similar to examples from class k. i.e. we assume all graphs from class k were sampled from the same distribution and try to estimate this distribution using various mathematical techniques.

        2. Graphon Mixup: Construct new classes W_α = λW_i + (1 − λ)W_j. i.e. do mixup on the estimated distributions from last step to get new distributions that would model some new (and made-up) classes.

        3. Graph Generation: New training samples {G_α1 , G_α2 , …}, {G_β1 , G_β2 , …}, … ∼ W_α, W_β , … i.e. sampled data points (graphs) from the made-up distributions.

        4. Label Mixup: Construct new labels yα = λyi + (1 − λ)yj i.e. find the labels for each graphon. What is the new class we made up a distribution for?

        Args:
            dataset (torch_geometric.InMemoryDataset): The base dataset to perform mixup on.
            specs (list[tuple]): List of tuples specifying mixup parameters. Each tuple contains:
                - label_class_i (int): Class label for the first graphon.
                - label_class_j (int): Class label for the second graphon.
                - mixup_fraction (float): Mixing fraction between the two graphons.
                - output_dim (int): Output dimensionality for graphon representation.
                - align_max_size (int): Maximum size for alignment.
                - nodes_param (float): Number of nodes in each graph is drawn from Geom(nodes_param)
                - num_samples_to_generate (int): Number of samples to generate for the mixup.
            estimation_method (str): Method for graphon estimation (e.g., 'usvt', 'sas', 'sba', etc.).
            args (dict): Additional arguments for the estimation method.
        """
        self.dataset = dataset
        self.specs = specs
        self._indices = dataset._indices
        self.estimation_method = estimation_method
        self._graphons = defaultdict(Graphon)
        self.new_graphs = []

        if self.specs is not None:
            self.process()

        # Represent the data for __getitem__
        self.graphs = dataset

    def process(self) -> None:
        """Process the dataset to perform mixup and generate new graph samples based on the specs.
        """
        class_graphs = self._split_class_graphs(self.dataset)

        for class_i, class_j, mixup_fraction, output_dim, align_max_size, nodes_param, num_samples in self.specs:
            # Generate or retrieve graphons for the classes
            class_i_graphon = self._graphons.get(class_i) or Graphon(
                graphs=class_graphs[class_i][1],
                padding=True,
                r=output_dim,
                label=class_i,
                align_max_size=align_max_size,
            )
            self._graphons[class_i] = class_i_graphon

            class_j_graphon = self._graphons.get(class_j) or Graphon(
                graphs=class_graphs[class_j][1],
                padding=True,
                r=output_dim,
                label=class_j,
                align_max_size=align_max_size,
            )
            self._graphons[class_j] = class_j_graphon

            # Perform mixup
            logging.info(
                "Creating new mixed graphon for clases: {class_i}, {class_j}")
            mixup_graphon, mixup_label = self._mixup(class_i_graphon,
                                                     class_j_graphon,
                                                     mixup_fraction)
            self._graphons[mixup_label] = mixup_graphon

            # Generate samples and add to new_graphs
            for _ in range(num_samples):

                new_sample, _ = mixup_graphon.generate(
                    np.random.geometric(nodes_param))

                edge_indices = np.array(np.nonzero(new_sample), dtype=np.int64)
                edge_index = torch.tensor(edge_indices, dtype=torch.long)

                data = Data(edge_index=edge_index,
                            y=torch.LongTensor([mixup_label]))
                data.num_nodes = new_sample.size[0]
                self.new_graphs.append(data)

    def _split_class_graphs(self, dataset: InMemoryDataset) -> list:
        """Split the dataset into graphs grouped by their class labels.
        Returns a list of tuples where each tuple contains a class label and corresponding graphs.

        Args:
            dataset (InMemoryDataset): The input dataset.

        :rtype: :class:`list`
        """
        y_list = [tuple(data.y.tolist()) for data in dataset]
        class_graphs = []

        for class_label in set(y_list):
            c_graph_list = [
                dataset[i] for i in range(len(y_list))
                if y_list[i] == class_label
            ]
            class_graphs.append((np.array(class_label), c_graph_list))

        return class_graphs

    def _mixup(self, class_i: Graphon, class_j: Graphon,
               mixup_fraction: float) -> tuple[Graphon, float]:
        """Perform mixup between two graphons.
        Returns a new mixed graphon and its label.

        Args:
            class_i (Graphon): The first graphon.
            class_j (Graphon): The second graphon.
            mixup_fraction (float): Mixing fraction between the two graphons.

        :rtype: (:class:`Graphon`, :class:`float`)
        """
        mixed_label = mixup_fraction * class_i._label + (
            1 - mixup_fraction) * class_j._label
        mixed_matrix = class_i._graphon * mixup_fraction + class_j._graphon * (
            1 - mixup_fraction)

        mixed_graphon = Graphon(
            graphs=[],
            padding=True,
            r=class_i.r,
            align_max_size=class_i.align_max_size,
            label=mixed_label,
            graphon=mixed_matrix,
        )
        return mixed_graphon, mixed_label

    def len(self) -> int:
        """Get the length of the dataset, including new generated graphs.

        Returns the total number of graphs in the dataset.

        :rtpe: :class:`int`
        """
        return len(self.dataset) + len(self.new_graphs)

    def __getitem__(self, idx: int) -> Data:
        """Retrieve a graph from the dataset by index.
        Returns the graph at the specified index.

        Args:
            idx (int): Index of the graph to retrieve. Negative indices are supported.

        :rtype: :class:`torch_geometric.data.Data`
        """
        if idx < 0:
            idx = len(self) + idx
            if idx < 0:
                raise ValueError("Index out of range.")

        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            return self.new_graphs[idx - len(self.dataset)]


from typing import List, Tuple

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# Methods for graphon estimation were adapted from the following repository:
# https://github.com/eomeragic1/g-mixup-reproducibility.git


class Graphon:
    r"""The Graphon object as defined in `G-Mixup` <https://arxiv.org/pdf/2202.07179>`_ paper.

    Graphons are functions W : [0, 1]² → [0, 1] representing the
    probability of an edge between two labeled vertices: W(i, j) = P[(i, j) ∈ E].
    This class estimates the graphon using input adjacency matrices.

    Args:
        graphs (list[np.ndarray]): List of adjacency matrices for estimation.
        padding (bool): Whether to align and pad graphs to the same size.
        r (int): Dimensionality of the output graphon (r x r matrix).
        align_max_size (int): Maximum number of graphs used for alignment.
        label (np.ndarray): Class labels for the graphs.
        estimation_method (str): Method for estimation ('lg' or 'usvt').
        graphon (np.ndarray, optional): Precomputed graphon matrix.
        **args: Additional arguments for estimation methods.
    """
    def __init__(self, graphs: List[np.ndarray], padding: bool, r: int,
                 align_max_size: int, label: np.ndarray,
                 estimation_method: str = "lg", graphon: np.ndarray = None,
                 **args):

        self.graphs = graphs
        self.padding = padding
        self.r = r
        self.estimation_method = estimation_method
        self._label = label
        self.align_max_size = align_max_size
        self._graphon = graphon

        if graphon is None:
            self._estimate()

    def generate(self, K: int) -> np.ndarray:
        """Generate a new random graph based on the estimated graphon.

        Args:
            K (int): Number of nodes in the generated graph.

        Returns:
            np.ndarray: Adjacency matrix of the generated graph.
        """
        nodes = np.random.uniform(size=(K, ))
        rounded_nodes = (nodes * self.r).astype(np.uint8)
        prob_vals = self._graphon[rounded_nodes[:, None], rounded_nodes]
        sampled_edges = (np.random.uniform(size=(K, K)) <= prob_vals)
        return sampled_edges, rounded_nodes

    def get_graphon(self) -> np.ndarray:
        """Retrieve the estimated graphon matrix.

        Returns:
            np.ndarray: The graphon matrix.
        """
        return self._graphon

    def _estimate(self):
        """Estimate the graphon matrix based on the chosen method ('lg' or 'usvt').
        """
        align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = self._align_graphs(
            self.graphs[:self.align_max_size], padding=self.padding, N=self.r)

        if self.estimation_method == 'lg':
            graphon = self._largest_gap(align_graphs_list, k=self.r,
                                        sum_graph=sum_graph)
        elif self.estimation_method == 'usvt':
            graphon = self._universal_svd(align_graphs_list,
                                          sum_graph=sum_graph)
        else:
            raise NotImplementedError(
                f"Invalid estimation_method: {self.estimation_method}")

        np.fill_diagonal(graphon, 0)
        self._graphon = graphon

    def _graph_numpy2tensor(self, graphs: List[np.ndarray]) -> torch.Tensor:
        """Convert a list of adjacency matrices from numpy to PyTorch tensor.

        Args:
            graphs (List[np.ndarray]): List of adjacency matrices.

        Returns:
            torch.Tensor: Tensor containing all adjacency matrices.
        """
        graph_tensor = np.array(graphs)
        return torch.from_numpy(graph_tensor).float()

    def _universal_svd(self, aligned_graphs: List[torch.Tensor],
                       threshold: float = 2.02,
                       sum_graph: torch.Tensor = None) -> np.ndarray:
        """Estimate the graphon using Universal Singular Value Thresholding (USVT).

        Args:
            aligned_graphs (List[torch.Tensor]): List of aligned adjacency matrices.
            threshold (float): Threshold for singular values.
            sum_graph (torch.Tensor, optional): Precomputed mean graph.

        Returns:
            np.ndarray: Estimated graphon matrix.
        """
        if sum_graph is None:
            aligned_graphs = self._graph_numpy2tensor(aligned_graphs)
            num_graphs = aligned_graphs.size(0)

            if num_graphs > 1:
                sum_graph = torch.mean(aligned_graphs, dim=0)
            else:
                sum_graph = aligned_graphs[0, :, :]

        num_nodes = sum_graph.size(0)
        u, s, v = torch.svd(sum_graph)
        singular_threshold = threshold * (num_nodes**0.5)
        s[s < singular_threshold] = 0
        graphon = u @ torch.diag(s) @ v.T
        graphon.clamp_(0, 1)
        return graphon.numpy()

    def _largest_gap(self, aligned_graphs: List[torch.Tensor], k: int,
                     sum_graph: torch.Tensor = None) -> np.ndarray:
        """Estimate the graphon using the largest gap method.

        Args:
            aligned_graphs (List[torch.Tensor]): List of aligned adjacency matrices.
            k (int): Number of blocks.
            sum_graph (torch.Tensor, optional): Precomputed mean graph.

        Returns:
            np.ndarray: Estimated graphon matrix.
        """
        if sum_graph is None:
            aligned_graphs = self._graph_numpy2tensor(aligned_graphs)

            if num_graphs > 1:
                sum_graph = torch.mean(aligned_graphs, dim=0)
            else:
                sum_graph = aligned_graphs[0, :, :]

        num_nodes = sum_graph.size(0)
        degree = torch.sum(sum_graph, dim=1)
        sorted_degree = degree / (num_nodes - 1)
        idx = torch.arange(0, num_nodes)

        diff_degree = sorted_degree[1:] - sorted_degree[:-1]
        _, index = torch.topk(diff_degree, k=k - 1)
        sorted_index, _ = torch.sort(index + 1)

        blocks = {}
        for b in range(k):
            if b == 0:
                blocks[b] = idx[:sorted_index[b]]
            elif b == k - 1:
                blocks[b] = idx[sorted_index[b - 1]:]
            else:
                blocks[b] = idx[sorted_index[b - 1]:sorted_index[b]]

        probability = torch.zeros(k, k)
        graphon = torch.zeros(num_nodes, num_nodes)
        for i in range(k):
            for j in range(k):
                rows, cols = blocks[i], blocks[j]
                subgraph = sum_graph[rows][:, cols]
                probability[i, j] = subgraph.mean()
                graphon[rows[:, None], cols] = probability[i, j]
        return graphon.numpy()

    def _align_graphs(
        self, graphs: List[Data], padding: bool = False, N: int = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int, torch.Tensor]:
        """Align multiple graphs by sorting their nodes by descending node degrees.

        This function sorts each graph's adjacency matrix by node degrees (from highest to lowest).
        Optionally, it can pad graphs to the same size or truncate them to a specified size (N).

        :param graphs: List of binary adjacency matrices represented as PyG Data objects.
        :param padding: Whether to pad graphs to the same size.
        :param N: If specified, truncates the graphs to size N (keeping the highest-degree nodes).
        :return:
            - aligned_graphs: List of aligned adjacency matrices as sparse tensors.
            - normalized_node_degrees: List of sorted, normalized node degree distributions.
            - max_num: Maximum number of nodes across all graphs.
            - min_num: Minimum number of nodes across all graphs.
            - avg_sum_graph: Average of the summed aligned adjacency matrices as a dense tensor.
        """
        num_nodes = [graph.num_nodes for graph in graphs]
        max_num = max(num_nodes)
        min_num = min(num_nodes)

        target_size = N if N else max_num
        sum_graph = np.zeros((target_size, target_size))

        aligned_graphs = []
        normalized_node_degrees = []

        for graph in graphs:
            num_i = graph.num_nodes
            adj = to_dense_adj(graph.edge_index)[0].numpy()

            # Calculate and normalize node degrees
            node_degree = (np.sum(adj, axis=0) + np.sum(adj, axis=1)) / 2
            node_degree /= np.sum(node_degree)

            # Sort nodes by degree (descending)
            idx = np.argsort(node_degree)[::-1]
            sorted_node_degree = node_degree[idx].reshape(-1, 1)
            sorted_graph = adj[idx][:, idx]

            # Apply padding or truncation
            if padding:
                normalized_node_degree = np.zeros((max_num, 1))
                normalized_node_degree[:num_i, :] = sorted_node_degree

                aligned_graph = np.zeros((max_num, max_num))
                aligned_graph[:num_i, :num_i] = sorted_graph
            else:
                normalized_node_degree = sorted_node_degree
                aligned_graph = sorted_graph

            truncated_degrees = normalized_node_degree[:target_size, :]
            truncated_graph = aligned_graph[:target_size, :target_size]

            # Append results
            normalized_node_degrees.append(truncated_degrees)
            sum_graph += truncated_graph
            aligned_graphs.append(
                dense_to_sparse(torch.from_numpy(truncated_graph))[0])

        avg_sum_graph = torch.from_numpy(sum_graph / len(graphs))
        return aligned_graphs, normalized_node_degrees, max_num, min_num, avg_sum_graph
