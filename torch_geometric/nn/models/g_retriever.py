from typing import List, Optional

import torch
import torch.nn as nn

from torch_geometric.nn.models import GAT
from torch_geometric.nn.text import LLM
from torch_geometric.nn.text.llm import (
    EOS,
    IGNORE_INDEX,
    max_new_tokens,
    max_txt_len,
)
from torch_geometric.utils import scatter


class GRetriever(nn.Module):
    r"""This GNN+LLM implementation is based on G-retriever.
    Original Paper: <https://arxiv.org/abs/2402.07630>`_.
    See `examples/llm_plus_gnn/g_retriever.py` for example usage.

    Args:
        llm_to_use (str): A string representing the huggingface model you
            want to use. This module has been tested for 'llama2' and 'gemma'.
            Other huggingface transformer models should work if you pass the
            correct name, see huggingface.co for details. If any issues occur
            please file an issue on
            https://github.com/pyg-team/pytorch_geometric
            and assign to puririshi98. (default: :obj:'llama2')
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
        gnn_in_channels (int): (default: 1024)
        gnn_hidden_channels (int): (default: 1024)
        gnn_out_channels (int): (default: 1024)
        num_gnn_layers (int): (default: 4)
        num_gnn_heads (int): Number of heads to use for BasicGNNs with the
        `heads` kwarg. (default: 4)
        mlp_hidden_dim (int): (default: 2048)
        mlp_out_dim (int): (default: 4096)
    """
    def __init__(
        self,
        llm_to_use='llama2',
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
    ) -> None:
        super().__init__()
        if 'llama' in llm_to_use.lower():
            self.llm_to_use = LLM('llama2', llm_dtype)
        elif 'gemma' in llm_to_use.lower():
            self.llm_to_use = LLM('gemma', llm_dtype)
        else:
            self.llm_to_use = LLM(llm_to_use, llm_dtype)
        self.llm_generator = self.llm_to_use.llm
        self.llm_dtype = llm_dtype
        if llm_use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            print("Training our LLM with LORA!")
            self.llm_generator = prepare_model_for_kbit_training(self.llm)
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
        print('Finished loading LLAMA!')

        self.graph_encoder = gnn_to_use(
            in_channels=gnn_in_channels,
            out_channels=gnn_out_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=num_gnn_layers,
            heads=num_gnn_heads,
            norm='batch_norm',
        ).to(self.llm_device)
        # For the MLP Projection
        mlp_hidden_dim = gnn_out_channels
        self.projector = nn.Sequential(
            nn.Linear(gnn_out_channels, mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim),
        ).to(self.llm_device)

        self.word_embedding = self.llm_to_use.word_embedding

    def encode_graphs(self, node_feat, edge_index, edge_attr, batch):
        x = node_feat.to(self.llm_device)
        edge_index = edge_index.long().to(self.llm_device)
        edge_attr = edge_attr.to(self.llm_device)
        n_embeds = self.graph_encoder(x, edge_index.long(), edge_attr)
        batch = batch.to(self.llm_device)
        # mean pooling
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
        batch_size, questions, context, eos_user_tokens, \
            bos_embeds, pad_embeds = self.llm_to_use.encode_inputs(question, additional_text_context) # noqa
        # encode labels
        labels = self.tokenizer(label, add_special_tokens=False)
        # encode training specific special token
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)

        # encode graphs
        graph_embeds = self.encode_graphs(node_feat, edge_index, edge_attr,
                                          batch)
        graph_embeds = self.projector(graph_embeds)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[
                i][:max_new_tokens] + eos_tokens.input_ids
            if additional_text_context is not None:
                input_ids = context.input_ids[
                    i][:max_txt_len] + questions.input_ids[
                        i] + eos_user_tokens.input_ids + label_input_ids
            else:
                input_ids = questions.input_ids[
                    i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            to_cat = [bos_embeds]
            if num_nodes_per_graph[i] != 0:
                to_cat.append(graph_embeds[i].unsqueeze(0))
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat([i.to(self.llm_device) for i in to_cat],
                                      dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX
                               ] * (inputs_embeds.shape[0] -
                                    len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad_length, 1).to(self.llm_device),
                batch_inputs_embeds[i].to(self.llm_device)
            ])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[
                i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(
            self.llm_device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(
            self.llm_device)

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
        max_out_tokens: Optional[int] = max_new_tokens,
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
        batch_size, questions, context, eos_user_tokens, \
            bos_embeds, pad_embeds = self.llm_to_use.encode_inputs(question, additional_text_context) # noqa
        # encode graphs
        graph_embeds = self.encode_graphs(node_feat, edge_index, edge_attr,
                                          batch)
        graph_embeds = self.projector(graph_embeds)

        batch_inputs_embeds = []
        batch_attention_mask = []
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        for i in range(batch_size):
            # Add bos & eos token
            if additional_text_context is not None:
                input_ids = context.input_ids[
                    i][:max_txt_len] + questions.input_ids[
                        i] + eos_user_tokens.input_ids
            else:
                input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.llm_device))
            to_cat = [bos_embeds]
            if num_nodes_per_graph[i] != 0:
                to_cat.append(graph_embeds[i].unsqueeze(0))
            to_cat.append(inputs_embeds)
            inputs_embeds = torch.cat([i.to(self.llm_device) for i in to_cat],
                                      dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0
                                       ] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds,
                                    dim=0).to(self.llm_device)
        attention_mask = torch.tensor(batch_attention_mask).to(
            self.llm_device)

        with self.llm_to_use.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'question': question,
            'desc': additional_text_context,
        }

    def print_trainable_params(self) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
