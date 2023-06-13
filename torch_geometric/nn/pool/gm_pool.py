from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

def argsort(batch: Tensor) -> Tuple[Tensor, Tensor]:
    idx = torch.argsort(batch)
    inverse = torch.arange(idx.shape[0], device = idx.device)[idx]
    return idx, inverse
    
@torch.no_grad()
def build_knn_graph(
    src_emb: Tenfor,
    dst_emb: Tenfor,
    src_batch: Tenfor,
    dst_batch: Tenfor,
    k: int
) -> Tenfor:
    src_idx, src_inverse = self.argsort(src_batch)
    dst_idx, dst_inverse = self.argsort(dst_batch)
    graph = knn(dst_emb[dst_idx], src_emb[src_idx], k, dst_batch[dst_idx], src_batch[src_idx])
    return torch.stack([src_inverse[graph[0]], dst_inverse[graph[1]]], dim = 0)

class GMPooling(torch.nn.Module):
    
    def __init__(
        self,
        resolution: float,
        min_size: int,
        build_bipartite_graph: Optional[bool] = False,
        build_super_graph: Optional[bool] = False,
        bipartite_k: Optional[int] = 5,
        super_k: Optional[int] = 5,
        momentum: Optional[float] = 0.95
    ):
        super().__init__()
        self.resolution = resolution
        self.min_size = min_size
        self.build_bipartite_graph = build_bipartite_graph
        self.build_super_graph = build_super_graph
        self.bipartite_k = bipartite_k
        self.super_k = super_k
        self.momentum = momentum
        self.model = GaussianMixture(2)
        if build_bipartite_graph:
            self.bipartite_graph_construction = DynamicGraphConstruction(bipartite_k, False, torch.exp)
        if build_super_graph:
            self.super_graph_construction = DynamicGraphConstruction(super_k, True, torch.sigmoid)
        self.register_buffer("score_cut", torch.tensor([0], dtype = torch.float), persistent = True)
        
    def _get_quadratic_coeff(
        self, 
        weight: float, 
        mean: float, 
        var: float
    ) -> Tuple[float, float, float]:
        sigma = np.sqrt(var)
        a = -0.5/sigma**2
        b = mean/sigma**2
        c = -0.5*mean**2/sigma**2 - np.log(sigma) + np.log(weight)
        return a, b, c

    def _solve_quadratic_eq(
        self,
        a: float,
        b: float,
        c: float
    ) -> float:
        if b**2 > 4*a*c:
            return torch.as_tensor((-b + np.sqrt(b**2 - 4*a*c))/(2*a), dtype = torch.float)
        else:
            return torch.as_tensor(-b/(2*a), dtype = torch.float)

    def _determine_cut(self) -> float:
        a1, b1, c1 = self._get_quadratic_coeff(self.model.weights_[0].item(), self.model.means_[0].item(), self.model.covariances_[0].item())
        a2, b2, c2 = self._get_quadratic_coeff(self.model.weights_[1].item(), self.model.means_[1].item(), self.model.covariances_[1].item())
        if self.model.means_[0][0] > self.model.means_[1][0]:
            return self._solve_quadratic_eq(a1-a2, b1-b2, c1-c2-self.resolution)
        else:
            return self._solve_quadratic_eq(a2-a1, b2-b1, c2-c1-self.resolution)
    
    def _get_clusters(
        self, 
        labels: Tensor, 
        batch: Tensor
    ) -> Tensor:
        _, inverse, counts = labels.unique(return_inverse = True, return_counts = True)
        noise_mask = counts[inverse] < self.min_size
        is_noise = ~scatter_add(~noise_mask, batch, dim = 0)
        labels[is_noise[batch]] = batch[is_noise[batch]] + labels.max() + 1
        noise_mask[is_noise[batch]] = False
        labels[~noise_mask] = labels[~noise_mask].unique(return_inverse = True)[1]
        labels[noise_mask] = -1
        return labels
    
    def forward(
        self, 
        emb: Tensor, 
        batch: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        
        emb = F.normalize(emb)
        edges = build_knn_graph(emb, emb, batch, batch, self.k)
        edges = edges[:, edges[0] != edges[1]]
        
        likelihood = (1-1e-6)*torch.einsum('ij,ij->i', emb[edges[0]], emb[edges[1]])
        likelihood = torch.atanh(likelihood)
            
        # GMM edge cutting
        inputs = likelihood.unsqueeze(1).cpu().detach().numpy()
        if inputs.size() > 10000:
            inputs = np.random.choice(inputs, 10000, replace = False)
        self.model.fit(inputs)
        cut = self.determine_cut()
        
        # Moving Average
        if self.training:
            self.score_cut = self.momentum*self.score_cut + (1-self.momentum)*cut
       
        # Connected Components
        mask = likelihood >= self.score_cut.to(likelihood.device)
        _, labels = connected_components(to_scipy_sparse_matrix(edges[:, mask], num_nodes = batch.shape[0]), directed = False)
        clusters = self._get_clusters(torch.as_tensor(labels, dtype = torch.long, device = emb.device), batch)
        
        # Compute centroids
        centroids = scatter_add(emb[clusters >= 0], clusters[clusters >= 0], dim = 0)
        centroids = F.normalize(centroids)
        centroids_batch = torch.zeros(centroids.shape[0], dtype = torch.long, device = centroids.device)\
                                    .scatter(0, clusters[clusters >= 0], batch[clusters >= 0])
        idxs = torch.argsort(centroids_batch)
        centroids, centroids_batch = centroids[idxs], centroids_batch[idxs]
        
        outputs = [centroids, centroids_batch]
        
        # Construct Bipartite Graph
        if self.build_bipartite_graph:
            outputs.extend(self.bipartite_graph_construction(emb, centroids, batch, centroids_batch))
            
        # Construct Super Graph
        if self.build_super_graph:
            outputs.extend(self.super_graph_construction(centroids, centroids, centroids_batch, centroids_batch))
        
        return outputs
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.resolution}, {self.min_size}, build_bipartite_graph = {self.build_bipartite_graph}, '
            f'build_super_graph = {self.build_super_graph}, '
            f'{f"bipartite_k = {self.bipartite_k}, " if self.build_bipartite_graph else ""}'
            f'{f"super_k = {self.super_k}, " if self.build_super_graph else ""}'
            f'momentum = {self.momentum})'
        )
        
class DynamicGraphConstruction(nn.Module):
    def __init__(
        self,
        k: int,
        sym: bool,
        weighting_function: Callable
    ):
        """
        weighting function is used to turn dot products into weights
        """
        super().__init__()
        
        self.weight_normalization = nn.BatchNorm1d(1)
        self.k = k
        self.sym = sym
        self.weighting_function = weighting_function

    def forward(
        self, 
        src_emb: Tensor , 
        dst_emb: Tensor, 
        src_batch: Tensor, 
        dst_batch: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Construct the Graph           
        graph = self.knn_search(src_emb, dst_emb, src_batch, dst_batch, self.k)
        if self.sym:
            graph = to_scipy_sparse_matrix(edges)
            graph, _ = from_scipy_sparse_matrix(graph + graph.T)
        
        # Compute bipartite attention
        likelihood = torch.einsum('ij,ij->i', src_emb[graph[0]], dst_emb[graph[1]]) 
        edge_weights_logits = self.weight_normalization(likelihood.unsqueeze(1)).squeeze() # regularize to ensure variance of weights
        edge_weights = self.weighting_function(edge_weights_logits)
    
        edge_weights = edge_weights/edge_weights.mean()
        edge_weights = edge_weights

        return graph, edge_weights
    


