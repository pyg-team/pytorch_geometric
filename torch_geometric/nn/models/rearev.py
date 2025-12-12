from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import softmax as pyg_softmax


class ReaRevReasoning(MessagePassing):
    r"""The reasoning layer used in the ReaRev model for iterative
    reasoning over knowledge graphs.

    This layer performs message passing with instruction-guided reasoning,
    where messages are weighted by node probabilities and separated into
    forward and inverse edge messages.

    Args:
        hidden_channels (int): The number of hidden channels.
        edge_attr_dim (int): The dimension of edge attributes.
        num_instructions (int): The number of instruction embeddings.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        hidden_channels: int,
        edge_attr_dim: int,
        num_instructions: int,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_instructions = num_instructions

        self.edge_proj = Linear(edge_attr_dim, hidden_channels)
        self.fuse_proj = Linear((2 * num_instructions + 1) * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.fuse_proj.weight)


    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        p: Tensor,
        edge_attr: Tensor,
        edge_type: Tensor,
        instruction: Tensor,
        size: Optional[tuple] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node features of shape :obj:`(num_nodes, hidden_channels)`.
            edge_index (torch.Tensor or SparseTensor): Edge connectivity.
            p (torch.Tensor): Node probabilities of shape :obj:`(num_nodes, 1)`.
            edge_attr (torch.Tensor): Edge attributes of shape
                :obj:`(num_edges, edge_attr_dim)`.
            edge_type (torch.Tensor): Edge types of shape :obj:`(num_edges,)`.
            instruction (torch.Tensor): Instruction embeddings of shape
                :obj:`(num_nodes, num_instructions, hidden_channels)`.
            size (tuple, optional): The size of the graph. (default: :obj:`None`)

        Returns:
            torch.Tensor: Updated node features of shape
                :obj:`(num_nodes, hidden_channels)`.
        """
        aggr_messages = self.propagate(
            edge_index,
            x=x,
            p=p,
            instruction=instruction,
            edge_type=edge_type,
            edge_attr=edge_attr,
        )
        aggr_messages = aggr_messages.flatten(start_dim=1)
        fused_out = self.fuse_proj(torch.cat([x, aggr_messages], dim=1))
        return F.relu(fused_out)


    def message(
        self,
        p_j: Tensor,
        instruction_i: Tensor,
        edge_type: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Constructs messages from node :obj:`j` to node :obj:`i`.

        Args:
            p_j (torch.Tensor): Node probabilities of source nodes.
            instruction_i (torch.Tensor): instruction embeddings
            edge_type (torch.Tensor): Edge types.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Concatenated forward and inverse messages.
        """
        edge_proj_out = self.edge_proj(edge_attr)
        # instruction_i is used to lift expanded instructions to the edge level
        # instruction_j could be used equivalently, only references the graph in the batch
        message = F.relu(instruction_i * edge_proj_out.unsqueeze(1))
        weighted = message * p_j.view(-1, 1, 1)

        is_inverse_edge = (edge_type == 1).view(-1,1,1)
        forward_msg = weighted.masked_fill(is_inverse_edge, 0)
        inverse_msg = weighted.masked_fill(~is_inverse_edge, 0)
        return torch.stack([forward_msg, inverse_msg], dim=1)


class ReaRev(torch.nn.Module):
    r"""The ReaRev model from the `"ReaRev: Adaptive Reasoning for Question
    Answering over Knowledge Graphs" <https://arxiv.org/abs/2210.13650>`_
    paper.

    This model performs iterative reasoning by generating instructions from
    question tokens and using them to guide breadth-first-style message passing
    over the graph. It iteratively updates node probabilities and refines
    instructions based on KG-aware context.

    Args:
        node_in_channels (int): Size of each input node feature.
        edge_attr_dim (int): Size of each edge attribute.
        hidden_channels (int): Size of each hidden representation.
        num_layers (int): Number of message passing layers.
        num_instructions (int): Number of instruction embeddings to generate.
        num_iterations (int): Number of reasoning iterations.
        question_embed_dim (int): Size of each question token embedding.
    """
    def __init__(
        self,
        node_in_channels: int,
        edge_attr_dim: int,
        hidden_channels: int,
        num_layers: int,
        num_instructions: int,
        num_iterations: int,
        question_embed_dim: int,
    ):
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

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
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

    def forward(
        self,
        question_tokens: Tensor,
        question_mask: Tensor,
        x: Tensor,
        edge_index: Adj,
        edge_type: Tensor,
        edge_attr: Tensor,
        seed_mask: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r"""Runs iterative question-guided reasoning over the graph.

        Args:
            question_tokens (torch.Tensor): Padded question tokens of shape
                :obj:`(batch_size, max_seq_len, question_embed_dim)`.
            question_mask (torch.Tensor): Mask indicating valid tokens of shape
                :obj:`(batch_size, max_seq_len)`, where :obj:`1` indicates a
                valid token and :obj:`0` indicates padding.
            x (torch.Tensor): Node features of shape
                :obj:`(num_nodes, node_in_channels)`.
            edge_index (torch.Tensor or SparseTensor): Edge connectivity.
            edge_type (torch.Tensor): Edge types of shape :obj:`(num_edges,)`.
            edge_attr (torch.Tensor): Edge attributes of shape
                :obj:`(num_edges, edge_attr_dim)`.
            seed_mask (torch.Tensor): Boolean mask indicating seed nodes of shape
                :obj:`(num_nodes,)`.
            batch (torch.Tensor): Batch vector assigning each node to a graph of
                shape :obj:`(num_nodes,)`.

        Returns:
            torch.Tensor: Node probabilities of shape :obj:`(num_nodes, 1)` after
            the final reasoning iteration, normalized per graph.
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

        # Initialize instructions
        generated_instructions = []
        prev_instruction = self.instruction_init.unsqueeze(0).expand(batch_size, -1)

        for k in range(self.num_ins):
            context = torch.cat(
                [
                    prev_instruction,
                    question_repr,
                    question_repr * prev_instruction,
                    question_repr - prev_instruction,
                ],
                dim=1,
            )
            instruction_query = self.W_k[k](context)

            token_interaction = instruction_query.unsqueeze(1) * question_proj_tokens
            token_scores = self.token_score_proj(token_interaction).squeeze(-1)
            token_scores = token_scores.masked_fill(question_mask == 0, -1e9)
            token_weights = F.softmax(token_scores, dim=1)

            # Here token_weights and question_proj_tokens are non-graph tensors so we can use torch.bmm
            instruction_k = torch.bmm(token_weights.unsqueeze(1), question_proj_tokens).squeeze(1)
            generated_instructions.append(instruction_k)
            prev_instruction = instruction_k

        instruction_stack = torch.stack(generated_instructions, dim=1)

        seed_weight = seed_mask.float().unsqueeze(-1)

        for t in range(self.num_iter):
            p = seed_weight

            # We extend the instruction stack to the batch level
            instruction_batch = instruction_stack[batch]
            for layer in self.layers:
                x = layer(
                    x=x,
                    p=p,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    instruction=instruction_batch,
                    edge_attr=edge_attr,
                )
                prob_logits = self.prob_proj(x)
                p = pyg_softmax(prob_logits, batch, dim=0)
            if t == self.num_iter - 1:
                break

            seed_rep = global_add_pool(x * seed_weight, batch)
            seed_rep_expanded = seed_rep.unsqueeze(1).expand(-1, self.num_ins, -1).reshape(
                -1, self.hidden_channels
            )
            instruction_flat = instruction_stack.view(-1, self.hidden_channels)
            fusion_inputs = torch.cat(
                [
                    instruction_flat,
                    seed_rep_expanded,
                    instruction_flat - seed_rep_expanded,
                    instruction_flat * seed_rep_expanded,
                ],
                dim=-1,
            )

            reset_out = self.reset_linear(fusion_inputs)
            gate_out = torch.sigmoid(self.gate_linear(fusion_inputs))
            instruction_stack = gate_out * reset_out + (1 - gate_out) * instruction_flat
            instruction_stack = instruction_stack.view(batch_size, self.num_ins, self.hidden_channels)

        return p