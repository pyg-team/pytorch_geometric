import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter

class GraphFeatureTokenizer(nn.Module):
    """
    A Graph Feature Tokenizer that converts graph structures into transformer-compatible tokens.
    
    This module implements the tokenization process described in the TokenGT paper (https://arxiv.org/abs/2207.02505),
    converting graph data (nodes, edges, and their features) into a sequence of tokens that can be
    processed by a transformer architecture.
    
    The tokenization process involves:
    1. Generating unique node identifiers using either random orthogonal features or Laplacian eigenvectors
    2. Combining node/edge features with positional and type embeddings
    3. Projecting the combined features to the transformer's hidden dimension
    4. Organizing tokens into a sequence with optional [graph] token

    Args:
        input_feat_dim (int): Input dimension of node/edge features
        hidden_dim (int): Output dimension for transformer input
        method (str): Method for generating node identifiers ('orf' or 'laplacian')
        d_p (int): Dimension of positional node identifiers
        d_e (int): Dimension of type embeddings
        use_graph_token (bool): Whether to prepend a [graph] token to the sequence
    """
    def __init__(self, input_feat_dim: int,
                 hidden_dim: int, method: str, d_p: int, d_e: int,
                 use_graph_token: bool = True):
        super(GraphFeatureTokenizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_p = d_p
        self.d_e = d_e
        self.method = method
        self.use_graph_token = use_graph_token

        # Learnable type embeddings to distinguish between nodes and edges
        self.E_V = nn.Parameter(torch.Tensor(1, d_e))  
        self.E_E = nn.Parameter(torch.Tensor(1, d_e))  

        # Input dimension includes: original features + 2 positional embeddings + type embedding
        self.input_dim = input_feat_dim + 2 * d_p + d_e
        # Linear projection to transformer's hidden dimension
        self.w_in = nn.Linear(self.input_dim, hidden_dim)

        # Optional learnable [graph] token embedding
        if self.use_graph_token:
            self.graph_token = nn.Parameter(torch.Tensor(1, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.E_V)
        nn.init.xavier_uniform_(self.E_E)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.zeros_(self.w_in.bias)
        if self.use_graph_token:
            nn.init.xavier_uniform_(self.graph_token)

    def forward(self, data: Batch) -> (torch.Tensor, torch.Tensor):
        """
        Transform a batch of graphs into sequences of tokens.

        Process:
        1. Extract graph components (nodes, edges, batch info)
        2. Generate node identifiers
        3. Combine features with positional and type embeddings
        4. Project to transformer dimension
        5. Arrange into sequences with attention masks

        Args:
            data (Batch): PyG batch containing multiple graphs

        Returns:
            tokens (torch.Tensor): Token embeddings [batch_size, seq_len, hidden_dim]
            attention_masks (torch.Tensor): Binary masks [batch_size, seq_len]
        """
        x = data.x 
        edge_index = data.edge_index 
        edge_attr = data.edge_attr  
        batch = data.batch 

        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_graphs = data.num_graphs

        # Generate structural node identifiers (positional embeddings)
        P = self.generate_node_identifiers(data)

        # Expand type embeddings to match batch size
        E_V = self.E_V.expand(num_nodes, self.d_e)  
        E_E = self.E_E.expand(num_edges, self.d_e)  

        # Combine node features with positional and type embeddings
        X_v = torch.cat([x, P, P, E_V], dim=1)  
        # Project to transformer dimension
        X_v_proj = self.w_in(X_v)  

        # Get positional embeddings for source and target nodes of each edge
        P_u = P[edge_index[0]]  
        P_v = P[edge_index[1]]  
        # Combine edge features with positional and type embeddings
        X_e = torch.cat([edge_attr, P_u, P_v, E_E], dim=1)  
        X_e_proj = self.w_in(X_e)  

        # Arrange tokens into sequences and create attention masks
        tokens, attention_masks = self.prepare_tokens(
            data, X_v_proj, X_e_proj, batch, edge_index
        )

        return tokens, attention_masks

    def generate_node_identifiers(self, data: Batch) -> torch.Tensor:
        """
        Generate structural node identifiers using either orthogonal random features (ORF)
        or Laplacian eigenvectors.

        Args:
            data (Batch): PyG batch containing multiple graphs

        Returns:
            torch.Tensor: Node identifiers matrix [num_nodes, d_p]
        """
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        device = x.device

        P_list = []

        for i in range(data.num_graphs):
            node_mask = (batch == i)
            num_nodes = node_mask.sum().item()

            if self.method == 'orf':
                # Generate orthogonal random features using QR decomposition
                G = torch.empty((num_nodes, num_nodes), device=device).normal_(mean=0, std=1)
                # Orthonormalize
                P, _ = torch.linalg.qr(G)  
                
                # Handle dimension mismatch
                if num_nodes < self.d_p:
                    # Pad if fewer nodes than desired dimensions
                    P = F.pad(P, (0, self.d_p - num_nodes), mode='constant', value=0)
                elif num_nodes > self.d_p:
                    perm = torch.randperm(num_nodes, device=device)
                    P = P[:, perm[:self.d_p]]
                P_list.append(P)
            
            elif self.method == 'laplacian':
                sub_edge_index, _ = subgraph(node_mask, edge_index, relabel_nodes=True)
                # Compute Laplacian eigenvectors
                P = self.compute_laplacian_eigenvectors(sub_edge_index, num_nodes)
                
                # Handle dimension mismatch
                if num_nodes < self.d_p:
                    P = F.pad(P, (0, self.d_p - num_nodes), mode='constant', value=0)
                elif num_nodes > self.d_p:
                    P = P[:, :self.d_p]
                P_list.append(P)
                
            else:
                raise ValueError("Invalid method for node identifiers. Choose 'orf' or 'laplacian'.")
        
        P = torch.cat(P_list, dim=0)
        assert P.size() == (x.size(0), self.d_p), f"Error while generating node identifiers: P has shape {P.size()}, expected {(x.size(0), self.d_p)}."
        return P

    def compute_laplacian_eigenvectors(self, edge_index, num_nodes: int) -> torch.Tensor:
        """
        Compute the eigenvectors of the graph Laplacian matrix.

        Args:
            edge_index (torch.Tensor): Edge connectivity
            num_nodes (int): Number of nodes in the graph

        Returns:
            torch.Tensor: Matrix of eigenvectors sorted by eigenvalue
        """
        device = edge_index.device
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj
        _, eigvecs = torch.linalg.eigh(laplacian)
        return eigvecs

    def prepare_tokens(self, data: Batch, X_v_proj: torch.Tensor, X_e_proj: torch.Tensor,
                       batch: torch.Tensor, edge_index: torch.Tensor):
        """
        Arrange node and edge tokens into sequences and create attention masks.

        The sequence structure is: [graph_token (optional) | node_tokens | edge_tokens]

        Args:
            data (Batch): PyG batch
            X_v_proj (torch.Tensor): Projected node features
            X_e_proj (torch.Tensor): Projected edge features
            batch (torch.Tensor): Batch assignments for nodes
            edge_index (torch.Tensor): Edge connectivity

        Returns:
            tokens (torch.Tensor): Token sequences [batch_size, seq_len, hidden_dim]
            attention_masks (torch.Tensor): Attention masks [batch_size, seq_len]
        """
        device = X_v_proj.device
        num_nodes = X_v_proj.size(0)
        num_edges = X_e_proj.size(0)
        num_graphs = data.num_graphs

        # Count nodes and edges per graph
        node_num = scatter(torch.ones(num_nodes, device=device), batch, dim=0, reduce='sum')
        edge_batch = batch[edge_index[0]]
        edge_num = scatter(torch.ones(num_edges, device=device), edge_batch, dim=0, reduce='sum')

        # Determine maximum sequence length
        max_node_num = int(node_num.max().item())
        max_edge_num = int(edge_num.max().item())
        seq_len = max_node_num + max_edge_num
        if self.use_graph_token:
            seq_len += 1

        # Initialize token tensors and attention masks
        tokens = torch.zeros((num_graphs, seq_len, self.hidden_dim), device=device)
        attention_masks = torch.zeros((num_graphs, seq_len), dtype=torch.bool, device=device)

        # Calculate cumulative sums for indexing
        node_ptr = torch.cat([node_num.new_zeros(1), node_num.cumsum(0)])
        edge_ptr = torch.cat([edge_num.new_zeros(1), edge_num.cumsum(0)])

        # Fill in tokens and attention masks for each graph
        for i in range(num_graphs):
            idx = 0
            # Add graph token if enabled
            if self.use_graph_token:
                tokens[i, idx] = self.graph_token
                attention_masks[i, idx] = True
                idx += 1

            # Add node tokens
            n_start = int(node_ptr[i].item())
            n_end = int(node_ptr[i + 1].item())
            n_nodes = n_end - n_start
            tokens[i, idx:idx + n_nodes] = X_v_proj[n_start:n_end]
            attention_masks[i, idx:idx + n_nodes] = True
            idx += max_node_num

            # Add edge tokens
            e_start = int(edge_ptr[i].item())
            e_end = int(edge_ptr[i + 1].item())
            n_edges = e_end - e_start
            tokens[i, idx:idx + n_edges] = X_e_proj[e_start:e_end]
            attention_masks[i, idx:idx + n_edges] = True

        return tokens, attention_masks
