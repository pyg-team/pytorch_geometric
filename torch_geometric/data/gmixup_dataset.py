import copy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from torch import diag, from_numpy, lt
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.utils import from_networkx, to_networkx


class GMixupDataset:
    r"""A wrapper class around a dataset that applies G-Mixup data augmentation
    to its base dataset.

    G-Mixup data augmentation technique is implemented as proposed in the paper
    "G-Mixup: Graph Data Augmentation for Graph Classification" by Xiaotian
    Han, Zhimeng Jiang, Ninghao Liu, Xia Hu in 2022.

    Args:
        base_dataset (Dataset): The dataset to apply G-Mixup to.
        log (bool, optional): Whether to print any console output while
            processing the dataset. (default: :obj:`True`)
        align_graphs (bool, optional): Whether to align the graphs by node
            degree before generating graphons. This is generally recommended to
            ensure graphon invariance to node ordering, but can be turned off
            for speed if it is known that the input graphs are already aligned.
            (default: :obj:`True`)
        threshold (float, optional): The threshold to use for singular value
            thresholding when generating graphons. Typically ranges from 2-3.
            (default: :obj:`2.02`)
        generate_graphons (bool, optional): Whether to generate graphons for
            the dataset during initialization or on-the-fly. Note that
            generating graphons for the dataset during initialization can be
            slow for large datasets. (default: :obj:`True`)
        sample_num (int, optional): How many graphs to sample from a class when
            approximating graphon for that class. Default value of -1 means to
            use all available graphs from that class. (default: :obj:`-1`)
    """
    def __init__(
        self,
        base_dataset: Dataset,
        log: bool = True,
        align_graphs: bool = True,
        threshold: float = 2.02,
        generate_graphons: bool = True,
        sample_num: int = -1,
    ) -> None:
        self.base_dataset = base_dataset
        self.log = log
        self.align_graphs = align_graphs
        self.threshold = threshold
        self.sample_num = sample_num

        self.graphs_by_class = [
            np.empty(0, dtype=int)
            for _ in range(self.base_dataset.num_classes)
        ]
        for i, data in enumerate(self.base_dataset):
            label = data.y.item()
            self.graphs_by_class[label] = np.append(
                self.graphs_by_class[label], i)

        self.max_node_count = int(
            np.max([data.num_nodes for data in self.base_dataset]))
        self.graphons = np.zeros((
            self.base_dataset.num_classes,
            self.max_node_count,
            self.max_node_count,
        ))
        self.graphons_generated = torch.zeros(self.base_dataset.num_classes,
                                              dtype=torch.bool)

        if generate_graphons:
            self.generate_graphons()
        else:
            if self.log:
                print("Graphons not generated during initialization."
                      "Graphons will be generated on-the-fly, or you can call"
                      "generate_graphons() to generate them all at once.\n")

    def generate_graphons(self) -> None:
        """Generates graphons for all classes in the dataset.

        If GMixupDataset was initialized with generate_graphons=False,
        generate_graphons() can be (optionally) manually called to generate
        graphons for all classes in the dataset. This can be useful if
        generate_graphons=False was used to speed up initialization, but
        graphons are needed later on.

        If generate_graphons is not called, graphons will be generated
        on-the-fly when needed by the generate_graphs.
        """
        if self.log:
            print("Generating graphons for "
                  f"{self.base_dataset.num_classes} classes...\n")

        for i in range(self.base_dataset.num_classes):
            if self.graphons_generated[i]:
                if self.log:
                    print(
                        f"Graphon for class {i} already generated, skipping..."
                    )
                continue
            else:
                self.__generate_graphon(i)

        return None

    def __align_graphs_by_degree(
        self, graph_adjs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
        num_nodes = [graph_adj.shape[0] for graph_adj in graph_adjs]
        max_num = max(num_nodes)
        min_num = min(num_nodes)

        aligned_adjs = []
        normalized_node_degrees = []
        for graph_adj in graph_adjs:
            curr_n = graph_adj.shape[0]
            node_degree = 0.5 * np.sum(graph_adj, axis=0) + 0.5 * np.sum(
                graph_adj, axis=1)
            node_degree /= np.sum(node_degree)
            perm = np.argsort(node_degree)  # ascending
            perm = perm[::-1]  # descending

            sorted_node_degree = node_degree[perm]
            sorted_node_degree = sorted_node_degree.reshape(-1, 1)

            sorted_graph = copy.deepcopy(graph_adj)
            sorted_graph = sorted_graph[perm, :]
            sorted_graph = sorted_graph[:, perm]

            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:curr_n, :] = sorted_node_degree
            aligned_adj = np.zeros((max_num, max_num))
            aligned_adj[:curr_n, :curr_n] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_adjs.append(aligned_adj)
        return aligned_adjs, normalized_node_degrees, max_num, min_num

    def __generate_graphon(self, class_idx: int) -> np.ndarray:
        import networkx as nx

        if self.graphons_generated[class_idx]:
            if self.log:
                print(f"Graphon for class {class_idx} already "
                      "generated, skipping...")
            return self.graphons[class_idx]

        num_graphs_of_class = len(self.graphs_by_class[class_idx])
        if self.sample_num > 0 and num_graphs_of_class > self.sample_num:
            class_sample = np.random.choice(self.graphs_by_class[class_idx],
                                            self.sample_num, replace=False)
        else:
            class_sample = self.graphs_by_class[class_idx]
        class_adj_mats = [
            nx.to_numpy_array(to_networkx(self.base_dataset[graph_index]))
            for graph_index in class_sample
        ]

        if self.align_graphs:
            aligned_graphs, normalized_node_degrees, max_num, min_num = (
                self.__align_graphs_by_degree(class_adj_mats))
            class_adj_mats = aligned_graphs

        graph_tensor_np = np.array(class_adj_mats)
        graph_tensor = from_numpy(graph_tensor_np).float()

        if self.log:
            print(f"Generating graphon for class {class_idx} with "
                  f"{graph_tensor.size(0)} graphs...")

        final_graphon = None
        count = 0
        for agg_graph_adj in tqdm(graph_tensor):
            # normalize adjacency matrix to be in interval [-1, 1]
            agg_graph_adj_normalized = 2 * agg_graph_adj - 1

            # svd graphon estimation \cite{Chatterjee}
            U, S, Vh = torch.linalg.svd(agg_graph_adj_normalized)

            num_nodes = agg_graph_adj.size(0)
            num_edges = torch.sum(agg_graph_adj == 1)
            proportion_edges = num_edges / (num_nodes**2) / 2

            scaled_thresh = self.threshold * (
                (num_nodes * proportion_edges)**0.5)
            S[lt(S, scaled_thresh)] = 0

            graphon = U @ diag(S) @ Vh

            graphon[graphon >= 1] = 1  # clip
            graphon[graphon <= -1] = -1  # clip
            # renormalize graphon to be in interval [0, 1]
            graphon = (graphon + 1) / 2
            graphon = torch.nn.functional.interpolate(
                graphon.unsqueeze(0).unsqueeze(0),
                size=(self.max_node_count, self.max_node_count),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            if final_graphon is None:
                final_graphon = graphon
            else:
                final_graphon = (final_graphon * count + graphon) / (count + 1)
            count += 1

        self.graphons[class_idx] = final_graphon.numpy()
        self.graphons_generated[class_idx] = True

        if self.log:
            print(f"Graphon for class {class_idx} generated.")
            print(f"graphon avg = {torch.mean(final_graphon)}, "
                  f"original avg = {torch.mean(graph_tensor)}. "
                  "These should be close.\n")

        return graphon.numpy()

    def generate_graphs(
        self,
        idx_1: Union[int, np.integer, IndexType],
        idx_2: Union[int, np.integer, IndexType],
        mixing_param: Union[float, np.float64, IndexType] = 0.5,
        K: Union[int, np.integer, IndexType] = 10,
        method: str = "random",
        size: Union[int, np.integer, IndexType] = 1,
    ) -> List[Data]:
        r"""Takes in a batch of graph label pairs and a mixing parameter λ, and
        returns the new synthetic graph(s) generated using G-Mixup.

        Args:
            idx_1 (int): Index of the first graph in the pair
            idx_2 (int): Index of the second graph in the pair
            mixing_param (float): The mixing parameter λ
            K (int): The number of nodes in the output synthetic graph(s)
            method (str): The method to use for generating the synthetic
            graph(s). Options are 'random' and 'uniform'.
            (default: :obj:`'random'`)
            size (int): The number of synthetic graphs to generate.
            (default: :obj:`1`)

        Returns:
            graphs (List[Data]): a list of the generated graphs
        """
        if not self.graphons_generated[idx_1]:
            if self.log:
                print(f"Graphon for class {idx_1} not yet generated, "
                      "generating...")
            self.__generate_graphon(idx_1)
        if not self.graphons_generated[idx_2]:
            if self.log:
                print(f"Graphon for class {idx_2} not yet generated, "
                      "generating...")
            self.__generate_graphon(idx_2)

        if self.log:
            print(f"Generating {size} synthetic graph(s) for indices {idx_1} "
                  f"and {idx_2} with mixing parameter {mixing_param} "
                  f"and {K} nodes...")

        graphs = []
        for i in range(size):
            graph = self.__generate_graph(idx_1, idx_2, mixing_param, K,
                                          method)
            graphs.append(graph)

        return graphs

    def __generate_graph(
        self,
        idx_1: int,
        idx_2: int,
        mixing_param: float = 0.5,
        K: int = 10,
        method: str = "random",
    ) -> Data:
        import networkx as nx

        graphon1 = self.graphons[idx_1]
        graphon2 = self.graphons[idx_2]
        mixed_graphon = mixing_param * graphon1 + (1 - mixing_param) * graphon2

        u_values = None
        u_values_index = None
        if method == "random":
            u_values = np.random.uniform(0, 1, K)
            u_values.sort()
            u_values_index = (u_values * mixed_graphon.shape[0]).astype(int)
        elif method == "uniform":
            u_values = np.linspace(0, 1, K, endpoint=False)
            u_values_index = (u_values * mixed_graphon.shape[0]).astype(int)
        else:
            raise ValueError("Invalid method")

        adjacency_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                p = mixed_graphon[u_values_index[i], u_values_index[j]]
                if np.random.rand() < p:
                    adjacency_matrix[i, j] = 1

        graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        graph_pyg = from_networkx(graph)

        if self.log:
            print("Synthetic graph(s) generated.\n")

        return graph_pyg

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.base_dataset, name)
