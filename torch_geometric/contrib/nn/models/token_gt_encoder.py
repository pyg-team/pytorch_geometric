import math
import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj, degree

class TokenGTEncoding(torch.nn.Module):
    def __init__(
        self,
        type_id_dim: int = 32,
        output_dim: int = 32,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super().__init__()
        self.type_id_dim = type_id_dim
        self.output_dim = output_dim
        self.node_id = None
        self.edge_id = None
        self.proj_matrix = None
        self.graph_embed = None

    def reset_parameters(self):
        pass

    def get_laplacian(self, edge_index: torch.Tensor) -> Tensor:
        adj_matrix = to_dense_adj(edge)
        n = adj_matrix.size(0)
        degree_matrix = torch.diag(degree(edge_index[0]))
        graph_laplacian = torch.eye(n) - degree_matrix**(-1/2) @ adj_matrix @ degree_matrix**(-1/2)
        l, v = torch.linalg.eig(graph_laplacian)
        return torch.linalg.inv(v)

    def get_orf(self) -> Tensor:
        n = nodes.size(0)
        random_gaussian = torch.nn.init.normal_((n, n))
        q, r = torch.linalg.qr(random_gaussian)
        return q

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: Torch.Tensor, use_laplace=True) -> Tensor:
        '''
        Args:
            nodes (torch.Tensor): The node feature matrix, of shape (n, embed_dim)
            edges (torch.Tensor): The edge feature matrix, of shape (n, embed_dim)
            edge_index (torch.Tensor): indices of edges, of shape (2, n)
            **kwargs (optional): Additional arguments passed to the GNN module.
        '''
        # Initialize matrices during forward pass because otherwise we don't know n
        with torch.no_grad(): # Freeze the node_id matrix
            if use_laplace:
                self.node_id = self.get_laplacian(edge_index)
            else:
                self.node_id = self.get_orf()
        self.type_id = torch.nn.init.normal_((2, self.type_id_dim))
        self.graph_embed = torch.nn.init.normal_((1, self.output_dim))
        # Augment node and edge tokens
        nodes_plus = torch.cat([nodes, self.node_id, self.node_id, self.type_id[:1]], dim=-1)
        u_embeds = self.node_id[:, edge_index[0]] #P_u to concat with edge embeddings
        v_embeds = self.node_id[:, edge_index[1]] #P_v to concat with edge embeddings
        edges_plus = torch.cat([edges, u_embeds, v_embeds, self.node_id, self.type_id[1:]], dim=-1)
        # Apply final concat and projection
        x_plus = torch.cat([nodes_plus, edge_plus], dim=0)
        self.proj_matrix = torch.nn.init.normal_((x_plus.size(1), self.output_dim))
        output = torch.cat([self.graph_embed, x_plus @ self.proj_matrix], dim=0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.output_dim})'