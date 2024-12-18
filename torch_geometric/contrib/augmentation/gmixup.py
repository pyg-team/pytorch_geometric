from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets.graph_generator import GraphGenerator


class GMixup(GraphGenerator):
    r"""The synthetic graph generator from the `"G-Mixup: Graph Data
    Augmentation for Graph Classification"
    <https://arxiv.org/abs/2202.07179>_` paper to generate mixup samples for
    graph classification.

    Args:
        dataset (torch_geometric.data.Dataset): A graph classification dataset
            for computing class graphons.
        method (str): The method to use for graphon estimation, either
            :obj:`"sorted_smooth"` (sorted smoothing method) or :obj:`"usvt"`
            (SVD-based thresholding method). Defaults to
            :obj:`"sorted_smooth"`.
    """
    def __init__(self, dataset: Dataset,
                 method: Literal['sorted_smooth', 'usvt'] = 'sorted_smooth', *,
                 sorted_smooth_h: int = 5, usvt_threshold: float = 2.02):
        super().__init__()
        self.sorted_smooth_h = sorted_smooth_h
        self.usvt_threshold = usvt_threshold

        # Partition into classes.
        class_graphs = {}
        for graph in dataset:
            if graph.y.numel() == 1:
                raise RuntimeError('graph labels must be one-hot vectors')
            label = tuple(graph.y.squeeze().tolist())
            if label not in class_graphs:
                class_graphs[label] = []
            class_graphs[label].append(graph)

        # Estimate graphons from classes.
        self.class_graphons = {}
        self.class_features = {}
        for label, graphs in class_graphs.items():
            features_list = [graph.x for graph in graphs]
            graphon, features = self.estimate_graphon(graphs, features_list,
                                                      method)
            self.class_graphons[label] = graphon
            self.class_features[label] = features

        # Pad graphons to the same size.
        num_nodes = max(f.shape[0] for f in self.class_features.values())
        for label, graphon in self.class_graphons.items():
            self.class_graphons[label] = self.pad_adjacency(graphon, num_nodes)
        for label, features in self.class_features.items():
            self.class_features[label] = self.pad_features(features, num_nodes)

        # Create mapping of idx -> label for fast class sampling.
        self.label_lookup = {
            i: l
            for i, l in enumerate(self.class_graphons.keys())
        }

    def __call__(self, interpolation_lambda: float) -> Data:
        r"""Equivalent to :obj:`self.sample`."""
        return self.sample(interpolation_lambda)

    def align_nodes(self, graph: Data, original_features: Tensor):
        r"""Aligns nodes and node features by sorting by degree."""
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes

        node_degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        sorted_indices = torch.argsort(node_degrees, descending=True)

        adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        adjacency_matrix[edge_index[0], edge_index[1]] = 1

        aligned_matrix = adjacency_matrix[sorted_indices][:, sorted_indices]
        aligned_features = original_features[sorted_indices]
        return aligned_matrix, aligned_features

    def estimate_graphon(
            self, graphs: List[Data], features_list: List[Tensor],
            method: str = "sorted_smooth") -> Tuple[Tensor, Tensor]:
        r"""Takes a set of graphs, returns an approximation of the graphon for
        that set of graphs. Uses one of two methods:
            "usvt": SVD-based thresholding method
            "sorted_smooth": Sorted smooth method.
        """
        aligned_adjacency = []
        aligned_features = []
        for graph, features in zip(graphs, features_list):
            aligned_mat, aligned_feat = self.align_nodes(graph, features)
            aligned_adjacency.append(aligned_mat)
            aligned_features.append(aligned_feat)

        # Pad adjacency and features to the maximum size.
        max_nodes = max(features.shape[0] for features in aligned_features)
        aligned_features = [
            self.pad_features(features, max_nodes)
            for features in aligned_features
        ]
        aligned_adjacency = [
            self.pad_adjacency(m, max_nodes) for m in aligned_adjacency
        ]
        # Use sum instead of stack+mean to reduce memory usage from copying.
        graphon_features = sum(aligned_features) / len(aligned_features)
        mean_adjacency = sum(aligned_adjacency) / len(aligned_adjacency)

        if method == "usvt":
            graphon = self.universal_svd(mean_adjacency,
                                         threshold=self.usvt_threshold)
            return graphon, graphon_features
        elif method == "sorted_smooth":
            graphon = self.sorted_smooth(mean_adjacency,
                                         h=self.sorted_smooth_h)
            return graphon, graphon_features
        else:
            raise ValueError(f"Unknown graphon estimation method: {method}")

    def generate(self, num_samples: int,
                 interpolation_range: Tuple[float, float]) -> List[Data]:
        r"""Generates synthetic graphs with aligned node features for data
        augmentation.

        Args:
            num_samples (int): Number of synthetic graphs to generate.
            interpolation_range (Tuple[float, float]): a low and high value
                for interpolation
        """
        synthetic_graphs = []
        low = interpolation_range[0]
        high = interpolation_range[1]
        for _ in range(num_samples):
            interp_lambda = low + (high - low) * torch.rand(1)
            graph = self.sample(interp_lambda)
            synthetic_graphs.append(graph)
        return synthetic_graphs

    def sample(self, interpolation_lambda: float) -> Data:
        r"""Generates one synthetic graph with aligned node features for data
        augmentation.

        Args:
            interpolation_lambda (float): Interpolation factor between
                graphons, features, and labels.
        """
        class1, class2 = np.random.choice(len(self.class_graphons), size=2,
                                          replace=False)
        label1 = self.label_lookup[class1]
        label2 = self.label_lookup[class2]
        graphon1 = self.class_graphons[label1]
        graphon2 = self.class_graphons[label2]
        features1 = self.class_features[label1]
        features2 = self.class_features[label2]

        mixed_graphon = self.mixup(graphon1, graphon2, interpolation_lambda)
        mixed_features = self.mixup(features1, features2, interpolation_lambda)
        label1 = torch.tensor(label1).to(mixed_graphon.device)
        label2 = torch.tensor(label2).to(mixed_graphon.device)
        mixed_label = self.mixup(label1, label2, interpolation_lambda)

        graph = self.generate_from_graphon(mixed_graphon, mixed_features)
        graph.y = mixed_label.unsqueeze(0)
        return graph

    def pad_features(self, features: Tensor, target_size: int) -> Tensor:
        r"""Pads the node features to the target size."""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        num_nodes, num_features = features.shape
        if num_nodes >= target_size:
            return features
        padded_features = torch.zeros((target_size, num_features),
                                      dtype=features.dtype)
        padded_features[:num_nodes, :] = features
        return padded_features

    def pad_adjacency(self, adj: torch.Tensor, target_size: int) -> Tensor:
        r"""Pads the adjacency matrix to the target size."""
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float)
        num_nodes = adj.shape[-1]
        padded_adj = torch.zeros((target_size, target_size), dtype=adj.dtype)
        padded_adj[:num_nodes, :num_nodes] = adj
        return padded_adj

    def mixup(self, tensor1: Tensor, tensor2: Tensor,
              interpolation_lambda: float) -> Tensor:
        r"""Interpolates between two tensors (ex. graphon, feature, label)
        based on an interpolation factor.
        """
        assert 0 <= interpolation_lambda <= 1, \
            "lambda should be in the range [0, 1]"
        return interpolation_lambda * tensor1 + \
            (1 - interpolation_lambda) * tensor2

    def generate_from_graphon(self, graphon: Tensor,
                              graphon_features: Tensor) -> Data:
        r"""Generates a synthetic graph from a graphon matrix and assigns
        node features.
        """
        num_nodes = graphon.shape[0]

        # generate adjacency matrix from graphon
        adjacency_matrix = (torch.rand(num_nodes, num_nodes) < graphon)
        adjacency_matrix = torch.triu(adjacency_matrix).to(torch.float32)

        # make the matrix symmetric
        adjacency_matrix += (adjacency_matrix.T -
                             torch.diag(torch.diag(adjacency_matrix)))

        edge_index = adjacency_matrix.nonzero(as_tuple=False).t()
        if not isinstance(graphon_features, torch.Tensor):
            graphon_features = torch.tensor(graphon_features,
                                            dtype=torch.float)

        synthetic_graph = Data(edge_index=edge_index)
        synthetic_graph.num_nodes = num_nodes
        synthetic_graph.x = graphon_features.clone().detach()

        return synthetic_graph

    def universal_svd(self, adj_matrix: Tensor,
                      threshold: float = 2.02) -> Tensor:
        r"""Estimate a graphon by universal singular value thresholding.
        Reference:
        Chatterjee, Sourav.
        "Matrix estimation by universal singular value thresholding."
        The Annals of Statistics 43.1 (2015): 177-214.
        """
        num_nodes = adj_matrix.size(0)
        u, s, v = torch.svd(adj_matrix)
        singular_threshold = threshold * (num_nodes**0.5)
        # Zero out singular values below threshold.
        s[s < singular_threshold] = 0
        graphon = u @ torch.diag(s) @ v.t()
        graphon = torch.clamp(graphon, min=0, max=1)
        return graphon

    def sorted_smooth(self, mean_adjacency: Tensor, h: int = 5) -> Tensor:
        r"""Implement the sorted_smooth method. This first averages the
        aligned graphs and then applies a block-averaging via a convolutional
        operation. Finally, it applies total variation denoising to produce a
        smooth graphon estimate.
        """
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            raise RuntimeError('sorted_smooth method requires scikit-image')
        num_nodes = mean_adjacency.shape[-1]
        mean_adjacency = mean_adjacency.reshape(1, 1, num_nodes, num_nodes)

        # Uniform kernel for block-averaging.
        kernel = torch.ones(1, 1, h, h) / (h**2)
        graphon = F.conv2d(mean_adjacency, kernel, padding=0, stride=h,
                           bias=None)
        graphon = graphon[0, 0, :, :].numpy()

        # Apply TV denoising (https://www.ipol.im/pub/art/2013/61/article.pdf).
        graphon = denoise_tv_chambolle(graphon, weight=h)

        graphon = torch.tensor(graphon, dtype=torch.float32)
        graphon = torch.clamp(graphon, min=0, max=1)
        return graphon
