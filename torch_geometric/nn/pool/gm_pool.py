from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn
from torch_geometric.utils import (
    scatter,
    to_undirected,
    to_scipy_sparse_matrix
)

try:
    from torch_cluster import knn
except ImportError:
    knn = None

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None


def argsort(batch: Tensor) -> Tuple[Tensor, Tensor]:
    idx = torch.argsort(batch)
    inverse = torch.arange(idx.shape[0], device=idx.device)[idx]
    return idx, inverse


@torch.no_grad()
def build_knn_graph(src_emb: Tensor, dst_emb: Tensor, src_batch: Tensor,
                    dst_batch: Tensor, k: int) -> Tensor:
    if knn is None:
            raise ImportError("torch_cluster is required by GMPooling.")
            
    src_idx, src_inverse = argsort(src_batch)
    dst_idx, dst_inverse = argsort(dst_batch)
    edge_index = knn(
        dst_emb[dst_idx],
        src_emb[src_idx],
        k,
        dst_batch[dst_idx],
        src_batch[src_idx]
    )
    return torch.stack(
        [src_inverse[edge_index[0]], dst_inverse[edge_index[1]]],
        dim=0
    )


class GMPooling(torch.nn.Module):
    r"""The Gaussion Mixture Pooling implementation from
    `"Hierarchical Graph Neural Networks for Particle Track Reconstruction"
    <https://arxiv.org/abs/2303.01640>`_ paper.

    In short, GMPooling pools a graph by computing edge scores and solving for
    the optimal score cut. The edge scores are defined as :math:`\tanh^{-1}
    (x_i\cdot x_j)`, where :math:`x_i` are normalized node embeddings. 
    Assuming that there are intra-cluster and inter-cluster edges, we fit a
    Gaussian Mixture Model of 2 components. The score cut is solved by finding
    the score where the difference in log likelihood of the two components 
    exceeds a hyperparameter :obj:`r` that is used to control the granularity
    of the pooling. Finally, the connected components that have more than 
    :obj:`min_size` nodes are regarded as the pooled nodes, which are called
    "supernodes" in the paper.

    GMPooling takes in node embeddings, graph edges, and batch indices and
    returns pooled embeddings (supernodes) and their batch indices.
    Optionally, GMPooling can also return bipartite graph than connects the
    original nodes and the newly created supernodes, together with edge
    weights that ensure the differentiability of the method, and the super
    graph on the supernodes and their edge weights.

    Args:
        r (int): Resolution which controls the granularity of the pooling.
        min_size (int): Minimum size for a connected component to be kept as
            supernode. Any component smaller than this will be treated as
            noise.
        build_bipartite_graph (bool): Whether to build bipartite
            graph or not. When set to :obj:`True`, edges and edge weights are
            returned.
        build_super_graph (bool): Whether to build super graph or not. When 
            set to :obj:`True`, edges and edge weights are returned.
        bipartite_k (int, optional): The connectivity of the bipartite graph
            being built. (default: :obj:`5`)
        super_k (int, optional): The connectivity of the super graph being
            built. (default: :obj:`5`)
        momentum (float, optional): The momentum of the exponential moving
            average used to track score cut. (default: :obj:`0.95`)
    """
    def __init__(self, r: float, min_size: int, build_bipartite_graph: bool,
                 build_super_graph: bool, bipartite_k: Optional[int] = 5,
                 super_k: Optional[int] = 5, 
                 momentum: Optional[float] = 0.95):
        super().__init__()
        
        if GaussianMixture is None:
            raise ImportError("sklearn is required by GMPooling.")
        
        self.r = r
        self.min_size = min_size
        self.build_bipartite_graph = build_bipartite_graph
        self.build_super_graph = build_super_graph
        self.bipartite_k = bipartite_k
        self.super_k = super_k
        self.momentum = momentum
        self.model = GaussianMixture(2)
        if build_bipartite_graph:
            self.bipartite_graph_construction = DynamicGraphConstruction(
                bipartite_k, False, torch.exp)
        if build_super_graph:
            # plus 1 since self-sycles will also be included
            self.super_graph_construction = DynamicGraphConstruction(
                super_k + 1, True, torch.sigmoid)
        self.register_buffer("score_cut", torch.tensor([0], dtype=torch.float),
                             persistent=True)

    def _get_quadratic_coeff(self, weight: float, mean: float,
                             var: float) -> Tuple[float, float, float]:
        """
        find the coefficients of the quadratic equation that defines score cut
        """
        sigma = np.sqrt(var)
        a = -0.5 / sigma**2
        b = mean / sigma**2
        c = -0.5 * mean**2 / sigma**2 - np.log(sigma) + np.log(weight)
        return a, b, c

    def _solve_quadratic_eq(self, a: float, b: float, c: float) -> Tensor:
        """
        solve the quadratic equation
        """
        if b**2 > 4 * a * c:
            return torch.as_tensor((-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a),
                                   dtype=torch.float)
        else:
            return torch.as_tensor(-b / (2 * a), dtype=torch.float)

    def _determine_cut(self) -> float:
        """
        extract the parameters of Gaussians and solve for score cut. Always
        pick the greater solution.
        """
        a1, b1, c1 = self._get_quadratic_coeff(
            self.model.weights_[0].item(), self.model.means_[0].item(),
            self.model.covariances_[0].item())
        a2, b2, c2 = self._get_quadratic_coeff(
            self.model.weights_[1].item(), self.model.means_[1].item(),
            self.model.covariances_[1].item())
        if self.model.means_[0][0] > self.model.means_[1][0]:
            return self._solve_quadratic_eq(a1 - a2, b1 - b2, c1 - c2 - self.r)
        else:
            return self._solve_quadratic_eq(a2 - a1, b2 - b1, c2 - c1 - self.r)

    def _get_clusters(self, labels: Tensor, batch: Tensor) -> Tensor:
        """
        filter out noises, i.e. components smaller than min_size.
        """
        _, inverse, counts = labels.unique(return_inverse=True,
                                           return_counts=True)
        noise_mask = counts[inverse] < self.min_size
        is_noise = ~scatter(~noise_mask, batch, dim=0, reduce="any")
        labels[is_noise[batch]] = batch[is_noise[batch]] + labels.max() + 1
        noise_mask[is_noise[batch]] = False
        labels[~noise_mask] = labels[~noise_mask].unique(
            return_inverse=True)[1]
        labels[noise_mask] = -1
        return labels

    def forward(
        self, x: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor],
               Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            x (Tensor): The node embeddings.
            edge_index (Tensor): The edge list of the graph.
            batch (Tensor): The batch indices.
        Returns
            centroids (Tensor): The embeddings of the pooled nodes
                (supernodes)
            centroids_batch (Tensor): The batch indices of the pooled nodes
                (supernodes)
            bipartite_graph (Optional, Tensor): The edge list of the bipartite
                graph built.
            bipartite_edge_weights (Optional, Tensor): The edge weights of
                the bipartite graph built.
            super_graph (Optional, Tensor): The edge list of the super graph
                built.
            super_edge_weights (Optional, Tensor): The edge weights of the
                super graph built.
        """
        # Normalize embeddings and remove self cycles
        x = F.normalize(x)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        # Compute likelihoods which is defined as atanh of cosine similarities
        likelihood = (1 - 1e-6) * torch.einsum('ij,ij->i', x[edge_index[0]],
                                               x[edge_index[1]])
        likelihood = torch.atanh(likelihood)

        # GMM edge cutting and compute moving average of score cut
        if self.training:
            inputs = likelihood.cpu().detach().numpy()
            if inputs.size > 10000:
                inputs = np.random.choice(inputs, 10000, replace=False)
            self.model.fit(inputs.reshape(-1, 1))
            cut = self._determine_cut()
            self.score_cut = self.momentum * self.score_cut + (
                1 - self.momentum) * cut

        # Connected Components
        mask = likelihood >= self.score_cut.to(likelihood.device)
        _, labels = connected_components(
            to_scipy_sparse_matrix(
                edge_index[:, mask],
                num_nodes=batch.shape[0]
            ),
            directed=False)
        clusters = self._get_clusters(
            torch.as_tensor(labels, dtype=torch.long, device=x.device),
            batch)

        # Compute centroids
        centroids = scatter(x[clusters >= 0], clusters[clusters >= 0], dim=0)
        centroids = F.normalize(centroids)
        centroids_batch = torch.zeros(centroids.shape[0], dtype=torch.long,
                                      device=centroids.device).scatter(
                                          0, clusters[clusters >= 0],
                                          batch[clusters >= 0])
        idxs = torch.argsort(centroids_batch)
        centroids, centroids_batch = centroids[idxs], centroids_batch[idxs]

        outputs = [centroids, centroids_batch]

        # Construct Bipartite Graph
        if self.build_bipartite_graph:
            outputs.extend(
                self.bipartite_graph_construction(x, centroids, batch,
                                                  centroids_batch))

        # Construct Super Graph
        if self.build_super_graph:
            outputs.extend(
                self.super_graph_construction(centroids, centroids,
                                              centroids_batch,
                                              centroids_batch))

        return outputs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.r}, {self.min_size},'
                f'build_bipartite_graph = {self.build_bipartite_graph}, '
                f'build_super_graph = {self.build_super_graph}, '
                f'bipartite_k = {self.bipartite_k}, '
                f'super_k = {self.super_k}, '
                f'momentum = {self.momentum})')


class DynamicGraphConstruction(nn.Module):
    def __init__(self, k: int, sym: bool, weighting_function: Callable):
        """
        Dynamic Graph Construction process. The knn graph is built and the
        edge are weighted by the value of weighting_function of cosine
        similarities. BatchNorm is used to normalize the cosine similarities
        to ensure variance of the weights. The edge weights are normalized
        to ensure that the mean of the weights is one.
        Args:
            k (int): the connectivity of the graph built
            sym (bool): to symmetrize the graph or not
            weighting_function (Callable): a function used to weight edges
        """
        super().__init__()

        self.weight_normalization = nn.BatchNorm1d(1)
        self.k = k
        self.sym = sym
        self.weighting_function = weighting_function

    def forward(self, src_x: Tensor, dst_x: Tensor, src_batch: Tensor,
                dst_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src_x (Tensor): The source node embeddings.
            dst_x (Tensor): The destination node embeddings.
            src_batch (Tensor): The source batch indices.
            dst_batch (Tensor): The destination batch indices.
        Returns
            graph (Tensor): The KNN graph built.
            edge_weights (Tensor): The edge weights computed.
        """
        # Construct the Graph
        edge_index = build_knn_graph(
            src_x,
            dst_x,
            src_batch,
            dst_batch,
            self.k
        )
        if self.sym:
            edge_index = to_undirected(edge_index)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        # Compute bipartite attention
        edge_weights = torch.einsum('ij,ij->i', src_x[edge_index[0]],
                                    dst_x[edge_index[1]])
        # Normalize to ensure variance of weights
        if len(edge_weights) > 1:
            edge_weights = self.weight_normalization(
                edge_weights.unsqueeze(1)).squeeze()

        edge_weights = self.weighting_function(edge_weights)

        edge_weights = edge_weights / edge_weights.mean()
        edge_weights = edge_weights

        return edge_index, edge_weights
