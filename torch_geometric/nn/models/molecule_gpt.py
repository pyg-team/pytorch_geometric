from typing import List, Optional

import torch
from torch import Tensor

from torch_geometric.nn.nlp.llm import LLM
from torch_geometric.utils import to_dense_batch


class MoleculeGPT(torch.nn.Module):
    r"""The MoleculeGPT model from the `"MoleculeGPT: Instruction
    Following Large Language Models for Molecular Property Prediction"
    <https://ai4d3.github.io/papers/34.pdf>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        gnn (torch.nn.Module): The GNN to use.
        mlp_out_channels (int, optional): The size of each graph embedding
            after projection. (default: :obj:`4096`)

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
        mlp_in_channels: int = 4928,
        mlp_out_channels: int = 2048,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.graph_encoder = graph_encoder.to(self.llm.device)
        self.smiles_encoder = smiles_encoder.to(self.llm.device)

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm
        # TODO: Add Q-Former layer

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_channels, mlp_in_channels),
            torch.nn.Sigmoid(),
            torch.nn.Linear(mlp_in_channels, mlp_out_channels),
        ).to(self.llm.device)

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
        batch_unique = batch.unique()
        batch_size = len(instructions)
        # 2D Graph Branch: [bs, node_len, d]
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)

        x_graph = self.graph_encoder(x, edge_index, edge_attr=edge_attr)
        x_graph = to_dense_batch(x_graph, batch)[0].view(batch_size, -1)
        # 1D SMILES Branch: [bs, seq_len, d]
        x_smiles = self.smiles_encoder.encode(
            smiles, output_device=self.llm.device).view(batch_size, -1)

        # TODO: Add Q-Former

        # Merge into LLMs
        x_cat = torch.cat([x_graph, x_smiles], dim=1)
        xs = self.projector(x_cat).split(1, dim=0)  # mock[bs, d]
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
    def inference(self):
        pass

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  graph={self.graph_encoder},\n'
                f'  smiles={self.smiles_encoder},\n'
                f')')
