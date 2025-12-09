from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import global_add_pool,MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class ReaRevReasoning(MessagePassing):

    def __init__(self, hidden_channels, edge_attr_dim, num_instructions, **kwargs):
        super().__init__(aggr='add')
        self.hidden_channels = hidden_channels
        self.num_instructions = num_instructions

        self.edge_proj = Linear(edge_attr_dim, hidden_channels)
        self.fuse_proj = Linear((2 * num_instructions + 1) * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.fuse_proj.weight)


    def forward(self, x, edge_index,p, edge_attr, edge_type, instruction, size=None):
        """
        Args:
            x: (E, H) - Node features
            p: (E, 1) - Node probabilities
            edge_index: (2, E) - Edge index
            edge_attr: (E, rel_dim) - Edge features
            edge_type: (E,) - Edge type
            instruction: (E, K, H) - Instruction embeddings
        """
        instruction_flat = instruction.flatten(start_dim=1)

        aggr_messages = self.propagate(
            edge_index,
            x=x,
            p=p,
            instruction_flat=instruction_flat,
            edge_type=edge_type,
            edge_attr=edge_attr,
        )
        fused_out = self.fuse_proj(torch.cat([x, aggr_messages], dim=1)) 
        return F.relu(fused_out) 


    def message(self, p_j, instruction_flat_i, edge_type, edge_attr):
        instruction_i = instruction_flat_i.view(-1,self.num_instructions,self.hidden_channels)
        edge_proj_out = self.edge_proj(edge_attr) 
        message = F.relu(instruction_i * edge_proj_out.unsqueeze(1))
        weighted_message = message * p_j.view(-1,1,1)
        message_flat = weighted_message.flatten(start_dim=1)

        is_inverse_edge = (edge_type == 1).view(-1, 1)

        forward_msg = message_flat.masked_fill(is_inverse_edge,0)
        inverse_msg = message_flat.masked_fill(~is_inverse_edge,0)
        return torch.cat([forward_msg, inverse_msg],dim=1)


class ReaRev(torch.nn.Module):
    def __init__(self, node_in_channels, edge_attr_dim, hidden_channels, num_layers, num_instructions, num_iterations, question_embed_dim):
        """
        Args:
            node_in_channels: (int) - Input node features dimension
            edge_in_channels: (int) - Input edge features dimension
            hidden_channels: (int) - Hidden channels dimension
            num_layers: (int) - Number of layers
            num_instructions: (int) - Number of instructions
            num_iterations: (int) - Number of iterations
            question_embed_dim: (int) - Question embedding dimension
        """
        super().__init__()
        self.node_dim = node_in_channels
        self.edge_attr_dim = edge_attr_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_ins = num_instructions
        self.num_iter = num_iterations

        self.node_input_proj = nn.Linear(node_in_channels, hidden_channels)
        self.question_proj = nn.Linear(question_embed_dim, hidden_channels)

        self.instruction_init = nn.Parameter(torch.randn(hidden_channels))
        self.instruction_score_vec = nn.Parameter(torch.randn(hidden_channels, 1))
        self.token_score_proj = nn.Linear(hidden_channels, 1)
        self.W_k = nn.ModuleList([
            nn.Linear(4 * hidden_channels, hidden_channels) for _ in range(num_instructions)
        ])

        self.layers = nn.ModuleList([
            ReaRevReasoning(
                hidden_channels=hidden_channels,
                edge_attr_dim=edge_attr_dim,
                num_instructions=num_instructions
            ) for _ in range(num_layers)
        ])

        self.reset_linear = nn.Linear(hidden_channels * 4, hidden_channels, bias=False)
        self.gate_linear = nn.Linear(hidden_channels * 4, hidden_channels, bias=False)


        self.prob_proj = nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.instruction_score_vec)
        nn.init.normal_(self.instruction_init)

        self.prob_proj.reset_parameters()
        self.question_proj.reset_parameters()
        self.node_input_proj.reset_parameters()
        self.token_score_proj.reset_parameters()
        
        nn.init.xavier_uniform_(self.reset_linear.weight)
        nn.init.xavier_uniform_(self.gate_linear.weight)

        for layer in self.W_k:
            layer.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, question_tokens, question_mask, x, edge_index, edge_type, edge_attr, seed_mask, batch):
        """
        Args:
            question_tokens: (Batch_Size, Max_Seq_Len, Embed_Dim) - Padded question tokens
            question_mask: (Batch_Size, Max_Seq_Len) - 1 for tokens, 0 for padding
            x: (Total_Nodes, Node_Dim) - Node features
            edge_index: (2, Total_Edges) - Edge index for each edge
            edge_type: (Total_Edges,) - Edge type for each edge
            edge_attr: (Total_Edges, Rel_Dim) - Edge features
            seed_mask: (Total_Nodes,) - Boolean mask indicating seed nodes
            batch: (Total_Nodes,) - Batch index for every node
        """
        batch_size = question_tokens.size(0)

        x = self.node_input_proj(x)
        x = F.relu(x)

        question_proj_tokens = self.question_proj(question_tokens)
        question_proj_tokens = F.relu(question_proj_tokens)

        mask_expanded = question_mask.unsqueeze(-1) 
        question_masked_sum = (question_proj_tokens * mask_expanded).sum(dim=1)
        question_token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        question_repr = question_masked_sum / question_token_counts

        generated_instructions = []

        prev_instruction = self.instruction_init.unsqueeze(0).expand(batch_size, -1)

        for k in range(self.num_ins):
            context = torch.cat([
                prev_instruction,
                question_repr,
                question_repr * prev_instruction,
                question_repr - prev_instruction
            ], dim=1)

            instruction_query = self.W_k[k](context)

            token_interaction = instruction_query.unsqueeze(1) * question_proj_tokens

            token_scores = self.token_score_proj(token_interaction).squeeze(-1)
            token_scores = token_scores.masked_fill(question_mask == 0, -1e9)

            token_weights = F.softmax(token_scores, dim=1)

            instruction_k = torch.bmm(token_weights.unsqueeze(1), question_proj_tokens).squeeze(1)

            generated_instructions.append(instruction_k)
            prev_instruction = instruction_k

        instruction_stack = torch.stack(generated_instructions, dim=1)

        p = torch.zeros(x.size(0),1, device=x.device)

        for t in range(self.num_iter):
            p = p.clone()
            if seed_mask is not None:
                p[seed_mask.bool()] = 1.0
            p = p / (global_add_pool(p, batch)[batch] + 1e-9)

            instruction_batch = instruction_stack[batch]
            for layer in self.layers:
                x = layer(x=x, 
                    p=p, 
                    edge_index=edge_index, 
                    edge_type=edge_type, 
                    instruction=instruction_batch, 
                    edge_attr=edge_attr
                )
                prob_logits = self.prob_proj(x)
                p = pyg_softmax(prob_logits,batch,dim=0)
            if t==self.num_iter-1:
                break

            batch_rep = global_add_pool(x*p, batch)

            batch_rep_expanded = batch_rep.unsqueeze(1).expand(-1, self.num_ins, -1).reshape(-1,self.hidden_channels)

            instruction_flat = instruction_stack.view(-1, self.hidden_channels)

            fusion_inputs = torch.cat([instruction_flat, batch_rep_expanded, instruction_flat - batch_rep_expanded, instruction_flat * batch_rep_expanded], dim=-1)

            reset_out = self.reset_linear(fusion_inputs)
            gate_out = torch.sigmoid(self.gate_linear(fusion_inputs))

            instruction_stack = gate_out * reset_out + (1 - gate_out) * instruction_flat
            instruction_stack = instruction_stack.view(batch_size, self.num_ins, self.hidden_channels)

        return p