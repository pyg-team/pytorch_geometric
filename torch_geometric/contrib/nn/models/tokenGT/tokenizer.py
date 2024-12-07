import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter

class GraphFeatureTokenizer(nn.Module):
    r"""
    Graph Feature Tokenizer for TokenGT.

    Args:
        node_feat_dim (int): Dimension of node features.
        edge_feat_dim (int): Dimension of edge features.
        hidden_dim (int): Input dimension of the transformer.
        method (str): Method to generate node identifiers ('orf' or 'laplacian').
        d_p (int): Dimension of node identifiers.
        d_e (int): Dimension of type identifiers.
        use_graph_token (bool): Whether to include the [graph] token.
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int, method: str, d_p: int, d_e: int,
                 use_graph_token: bool = True):
        super(GraphFeatureTokenizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_p = d_p
        self.d_e = d_e
        self.method = method
        self.use_graph_token = use_graph_token

        # Type identifiers (trainable embeddings)
        self.E_V = nn.Parameter(torch.Tensor(1, d_e))
        self.E_E = nn.Parameter(torch.Tensor(1, d_e))

        # Projection matrix w_in
        self.input_dim = node_feat_dim + 2 * d_p + d_e
        self.w_in = nn.Linear(self.input_dim, hidden_dim)

        # Embedding for [graph] token
        if self.use_graph_token:
            self.graph_token = nn.Parameter(torch.Tensor(1, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.E_V)
        nn.init.xavier_uniform_(self.E_E)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.zeros_(self.w_in.bias)
        if self.use_graph_token:
            nn.init.xavier_uniform_(self.graph_token)

    def forward(self, data: Batch) -> (torch.Tensor, torch.Tensor):
        r"""
        Forward pass of the GraphFeatureTokenizer.

        Args:
            data (Batch): PyG Batch object containing graph data.

        Returns:
            tokens (torch.Tensor): Token embeddings, shape [batch_size, seq_len, hidden_dim].
            attention_masks (torch.Tensor): Attention masks, shape [batch_size, seq_len].
        """
        x = data.x  
        edge_index = data.edge_index 
        edge_attr = data.edge_attr  
        batch = data.batch  

        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_graphs = data.num_graphs

        P = self.generate_node_identifiers(data)

        E_V = self.E_V.expand(num_nodes, self.d_e) 
        E_E = self.E_E.expand(num_edges, self.d_e) 

        X_v = torch.cat([x, P, P, E_V], dim=1)  

        X_v_proj = self.w_in(X_v)  

        P_u = P[edge_index[0]]  
        P_v = P[edge_index[1]] 
        X_e = torch.cat([edge_attr, P_u, P_v, E_E], dim=1)  

        # Project edge features
        X_e_proj = self.w_in(X_e)  

        tokens, attention_masks = self.prepare_tokens(
            data, X_v_proj, X_e_proj, batch, edge_index
        )

        return tokens, attention_masks

    def generate_node_identifiers(self, data: Batch) -> torch.Tensor:
        """Generate node identifiers using the specified method."""
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        num_nodes = x.size(0)
        device = x.device

        if self.method == 'orf':
            P = torch.empty((self.d_p, self.d_p), device=device).normal_(mean=0, std=1)
            # Orthonormalize
            P, _ = torch.linalg.qr(P)  
            P = P[:num_nodes, :]
        elif self.method == 'laplacian':
            P_list = []
            for i in range(data.num_graphs):
                node_mask = (batch == i)
                n_nodes = node_mask.sum().item()
                sub_edge_index, _ = subgraph(node_mask, edge_index, relabel_nodes=True)
                P_i = self.compute_laplacian_eigenvectors(sub_edge_index, n_nodes)
                P_list.append(P_i)
            P = torch.cat(P_list, dim=0) 
        else:
            raise ValueError("Invalid method for node identifiers. Choose 'orf' or 'laplacian'.")

        return P

    def compute_laplacian_eigenvectors(self, edge_index, num_nodes: int) -> torch.Tensor:
        """Compute Laplacian eigenvectors for node identifiers."""
        device = edge_index.device
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
        eigvecs = eigvecs[:, 1:self.d_p+1]
        return eigvecs

    def prepare_tokens(self, data: Batch, X_v_proj: torch.Tensor, X_e_proj: torch.Tensor,
                       batch: torch.Tensor, edge_index: torch.Tensor):
        """Prepare tokens and attention masks."""
        device = X_v_proj.device
        num_nodes = X_v_proj.size(0)
        num_edges = X_e_proj.size(0)
        num_graphs = data.num_graphs

        node_num = scatter(torch.ones(num_nodes, device=device), batch, dim=0, reduce='sum')  
        edge_batch = batch[edge_index[0]] 
        edge_num = scatter(torch.ones(num_edges, device=device), edge_batch, dim=0, reduce='sum') 

        max_node_num = node_num.max().item()
        max_edge_num = edge_num.max().item()

        seq_len = max_node_num + max_edge_num
        if self.use_graph_token:
            seq_len += 1  

        tokens = torch.zeros((num_graphs, seq_len, self.hidden_dim), device=device)
        attention_masks = torch.zeros((num_graphs, seq_len), dtype=torch.bool, device=device)

        node_ptr = torch.cat([node_num.new_zeros(1), node_num.cumsum(0)])
        edge_ptr = torch.cat([edge_num.new_zeros(1), edge_num.cumsum(0)]) 

        for i in range(num_graphs):
            idx = 0
            if self.use_graph_token:
                tokens[i, idx] = self.graph_token
                attention_masks[i, idx] = True
                idx += 1

            n_start = node_ptr[i].item()
            n_end = node_ptr[i + 1].item()
            n_nodes = n_end - n_start
            tokens[i, idx:idx + n_nodes] = X_v_proj[n_start:n_end]
            attention_masks[i, idx:idx + n_nodes] = True
            idx += max_node_num 

            e_start = edge_ptr[i].item()
            e_end = edge_ptr[i + 1].item()
            n_edges = e_end - e_start
            tokens[i, idx:idx + n_edges] = X_e_proj[e_start:e_end]
            attention_masks[i, idx:idx + n_edges] = True

        return tokens, attention_masks