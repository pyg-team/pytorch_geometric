from typing import List, Optional

import torch
import torch.nn as nn

from torch_geometric.nn.models import GAT
from torch_geometric.nn.nlp.llm import LLM, MAX_NEW_TOKENS
from torch_geometric.utils import scatter


class GRetriever(nn.Module):
    r"""This GNN+LLM implementation is based on G-retriever.
    Original Paper: <https://arxiv.org/abs/2402.07630>`_.

    Args:
        llm_to_use (str): A string representing the huggingface model you
            want to use. (default: :obj:`"meta-llama/Llama-2-7b-chat-hf"`)
        llm_use_lora (bool): use LORA from peft for training the LLM. see
            https://huggingface.co/docs/peft/en/index for details.
        llm_dtype (torch.dtype): The dtype to use for the LLM.
            (default :obj: `torch.bloat16`)
        num_llm_params (int): An integer representing how many params your
            huggingface transformer model has, in billions. This is used to
            automatically allocate the number of gpus needed, given the
            available GPU memory of your GPUs (default :obj:`7`)
        gnn_to_use (BasicGNN): Please pass a valid model that extends
            torch_geometric.nn.models.basic_gnn.BasicGNN. (default: :obj:`GAT`)
        gnn_in_channels (int): (default: :obj:`1024`)
        gnn_hidden_channels (int): (default: obj:`1024`)
        gnn_out_channels (int): (default: :obj:`1024`)
        num_gnn_layers (int): (default: :obj:`4`)
        num_gnn_heads (int): Number of heads to use for BasicGNNs with the
            :obj:`heads` kwarg. (default: :obj:`4`)
        mlp_hidden_dim (int): (default: :obj:`2048`)
        mlp_out_dim (int): (default: :obj:`4096`)

    .. warning::
        This module has been tested with the following Hugging Face models

        - :obj:`llm_to_use="meta-llama/Llama-2-7b-chat-hf"`
        - :obj:`llm_to_use="google/gemma-7b"`

        and may not work with other models. See other models at `Hugging Face
        Models <https://huggingface.co/models>`_ and let us know if you
        encounter any issue by submitting a GitHub issue.

    .. note::
        See `examples/llm_plus_gnn/g_retriever.py` for example usage.
    """
    def __init__(
        self,
        llm_to_use='meta-llama/Llama-2-7b-chat-hf',
        llm_use_lora: bool = False,
        llm_dtype=torch.bfloat16,
        num_llm_params: int = 7,
        gnn_to_use=GAT,
        gnn_in_channels: int = 1024,
        gnn_hidden_channels: int = 1024,
        gnn_out_channels: int = 1024,
        num_gnn_layers: int = 4,
        num_gnn_heads: int = 4,
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 4096,
        mlp_out_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.llm_to_use = LLM(llm_to_use, num_llm_params, llm_dtype)
        self.llm_generator = self.llm_to_use.llm
        self.llm_dtype = llm_dtype
        if llm_use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            self.llm_generator = prepare_model_for_kbit_training(
                self.llm_generator)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_generator = get_peft_model(self.llm_generator, config)
        self.llm_device = self.llm_to_use.llm_device
        self.tokenizer = self.llm_to_use.tokenizer
        try:
            self.graph_encoder = gnn_to_use(
                in_channels=gnn_in_channels,
                out_channels=gnn_out_channels,
                hidden_channels=gnn_hidden_channels,
                num_layers=num_gnn_layers,
                heads=num_gnn_heads,
                norm='batch_norm',
            ).to(self.llm_device)
        except:  # noqa: E722
            # to handle gnns that do not have `heads` param
            self.graph_encoder = gnn_to_use(
                in_channels=gnn_in_channels,
                out_channels=gnn_out_channels,
                hidden_channels=gnn_hidden_channels,
                num_layers=num_gnn_layers,
                norm='batch_norm',
            ).to(self.llm_device)
        # For the MLP Projection
        mlp_hidden_dim = gnn_out_channels
        self.projector = nn.Sequential(
            nn.Linear(gnn_out_channels, mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim*mlp_out_tokens),
        ).to(self.llm_device)

        self.mlp_out_tokens = mlp_out_tokens

        self.word_embedding = self.llm_to_use.word_embedding

    def encode_graphs(self, node_feat, edge_index, edge_attr, batch):
        x = node_feat.to(self.llm_device)
        edge_index = edge_index.long().to(self.llm_device)
        edge_attr = edge_attr.to(self.llm_device)
        n_embeds = self.graph_encoder(x, edge_index.long(), edge_attr)
        batch = batch.to(self.llm_device)
        g_embeds = scatter(n_embeds, batch, dim=0, reduce='mean')
        return g_embeds

    def forward(
        self,
        question: List[str],
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        label: List[str],
        edge_attr: Optional[torch.Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
    ):
        r"""Forward pass.

        Args:
            question (List[str]): The questions/prompts.
            node_feat (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            ptr (torch.Tensor): The pointer vector, denoting the
                boundaries between examples in the batch.
            label (List[str]): The answers/labels.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN being used). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
        """
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        graph_embeds = self.encode_graphs(node_feat, edge_index, edge_attr,
                                          batch)
        projected_graph_embeds = self.projector(graph_embeds)
        graph_embeds = []
        for i, num_nodes_in_graph_i in enumerate(num_nodes_per_graph):
            if num_nodes_in_graph_i == 0:
                graph_embeds.append(None)
            else:
                graph_embeds.append(projected_graph_embeds[i].unsqueeze(0))

        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm_to_use._get_embeds(question, additional_text_context,
                                        graph_embeds, label)

        with self.llm_to_use.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    @torch.no_grad()
    def inference(
        self,
        question: List[str],
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        additional_text_context: Optional[List[str]] = None,
        max_out_tokens: Optional[int] = MAX_NEW_TOKENS,
    ):
        r"""Inference.

        Args:
            question (List[str]): The questions/prompts.
            node_feat (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
            ptr (torch.Tensor): The pointer vector, denoting the
                boundaries between examples in the batch.
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the GNN being used). (default: :obj:`None`)
            additional_text_context (List[str], optional): Additional context
                to give to the LLM, such as textified knowledge graphs.
            max_out_tokens (int, optional): How many tokens for the LLM to
                generate. (default: {32})
        """
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        graph_embeds = self.encode_graphs(node_feat, edge_index, edge_attr,
                                          batch)
        graph_embeds = [
            (embed.unsqueeze(0) if num_nodes_per_graph[i] != 0 else None)
            for i, embed in enumerate(self.projector(graph_embeds))
        ]
        inputs_embeds, attention_mask, _ = self.llm_to_use._get_embeds(
            question, additional_text_context, graph_embeds)
        with self.llm_to_use.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
