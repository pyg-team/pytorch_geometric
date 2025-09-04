from typing import List, Optional

import torch
from torch import Tensor

from torch_geometric.nn.attention import QFormer
from torch_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS
from torch_geometric.utils import to_dense_batch


def pad_or_truncate(embeddings: Tensor, max_seq_len: int,
                    padding_value: int = 0) -> Tensor:
    batch_size, current_seq_len, d = embeddings.size()

    if current_seq_len > max_seq_len:
        return embeddings[:, :max_seq_len, :]
    elif current_seq_len < max_seq_len:
        pad_tensor = torch.full((batch_size, max_seq_len - current_seq_len, d),
                                padding_value, dtype=embeddings.dtype,
                                device=embeddings.device)
        return torch.cat([embeddings, pad_tensor], dim=1)
    else:
        return embeddings


class MoleculeGPT(torch.nn.Module):
    r"""The MoleculeGPT model from the `"MoleculeGPT: Instruction
    Following Large Language Models for Molecular Property Prediction"
    <https://ai4d3.github.io/papers/34.pdf>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        graph_encoder (torch.nn.Module): Encode 2D molecule graph.
        smiles_encoder (torch.nn.Module): Encode 1D SMILES.
        mlp_out_channels (int, optional): The size of each embedding
            after qformer encoding. (default: :obj:`32`)
        max_tokens (int, optional): Max output tokens of 1D/2D encoder.
            (default: :obj:`20`)

    .. warning::
        This module has been tested with the following HuggingFace models

        * :obj:`llm_to_use="lmsys/vicuna-7b-v1.5"`

        and may not work with other models. See other models at `HuggingFace
        Models <https://huggingface.co/models>`_ and let us know if you
        encounter any issues.

    .. note::
        For an example of using :class:`MoleculeGPT`, see
        `examples/llm/molecule_gpt.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/molecule_gpt.py>`_.
    """
    def __init__(
        self,
        llm: LLM,
        graph_encoder: torch.nn.Module,
        smiles_encoder: torch.nn.Module,
        mlp_out_channels: int = 32,
        max_tokens: Optional[int] = 20,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.graph_encoder = graph_encoder.to(self.llm.device)
        self.smiles_encoder = smiles_encoder.to(self.llm.device)

        self.graph_qformer = QFormer(
            input_dim=self.graph_encoder.nn[-1].out_features,
            hidden_dim=mlp_out_channels,
            output_dim=mlp_out_channels,
            num_heads=4,
            num_layers=2,
        ).to(self.llm.device)

        self.smiles_qformer = QFormer(
            input_dim=self.smiles_encoder.model.pooler.dense.out_features,
            hidden_dim=mlp_out_channels,
            output_dim=mlp_out_channels,
            num_heads=4,
            num_layers=2,
        ).to(self.llm.device)

        self.max_tokens = max_tokens

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm

        # LLMs
        in_dim = 2 * mlp_out_channels * max_tokens
        out_dim = self.llm.llm.model.embed_tokens.embedding_dim
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_dim, out_dim),
        ).to(self.llm.device)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
    ) -> Tensor:
        batch_size = len(smiles)
        # 2D Graph Branch: [bs, node_len, d]
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)

        x_graph = self.graph_encoder(x, edge_index, edge_attr=edge_attr)
        x_graph = to_dense_batch(x_graph, batch)[0]
        out_graph = self.graph_qformer(x_graph)
        out_graph = pad_or_truncate(out_graph, max_seq_len=self.max_tokens,
                                    padding_value=0)
        out_graph = out_graph.view(batch_size, -1)

        # 1D SMILES Branch: [bs, seq_len, d]
        x_smiles = self.smiles_encoder.encode(smiles,
                                              output_device=self.llm.device)
        out_smiles = self.smiles_qformer(x_smiles)
        out_smiles = pad_or_truncate(out_smiles, max_seq_len=self.max_tokens,
                                     padding_value=0)
        out_smiles = out_smiles.view(batch_size, -1)

        # Merge into LLMs
        x_cat = torch.cat([out_graph, out_smiles], dim=1)
        return x_cat

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
        instructions: List[str],
        label: List[str],
        additional_text_context: Optional[List[str]] = None,
    ):
        x = self.encode(x, edge_index, batch, edge_attr, smiles)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        batch_unique = batch.unique()
        batch_size = len(instructions)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(instructions, additional_text_context, xs,
                                 label)

        with self.llm.autocast_context:
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
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
        instructions: List[str],
        additional_text_context: Optional[List[str]] = None,
        max_out_tokens: Optional[int] = MAX_NEW_TOKENS,
    ):
        x = self.encode(x, edge_index, batch, edge_attr, smiles)
        x = self.projector(x)
        xs = x.split(1, dim=0)

        # Handle questions without node features:
        batch_unique = batch.unique()
        batch_size = len(instructions)
        if len(batch_unique) < batch_size:
            xs = [
                xs[i] if i in batch_unique else None for i in range(batch_size)
            ]

        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            instructions, additional_text_context, xs)

        bos_token = self.llm.tokenizer(
            BOS,
            add_special_tokens=False,
        ).input_ids[0]

        with self.llm.autocast_context:
            outputs = self.llm_generator.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_out_tokens,
                attention_mask=attention_mask,
                bos_token_id=bos_token,
                use_cache=True  # Important to set!
            )

        return self.llm.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  graph={self.graph_encoder.__class__.__name__},\n'
                f'  smiles={self.smiles_encoder},\n'
                f')')
