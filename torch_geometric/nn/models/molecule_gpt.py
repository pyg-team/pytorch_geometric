from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.nlp.llm import LLM
from torch_geometric.utils import scatter


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
        mlp_out_channels: int = 4096,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.graph_encoder = graph_encoder.to(self.llm.device)
        self.smiles_encoder = smiles_encoder.to(self.llm.device)

        self.word_embedding = self.llm.word_embedding
        self.llm_generator = self.llm.llm
        mlp_hidden_channels = self.gnn.out_channels
        # TODO: Add Q-Former layer
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_channels, mlp_out_channels),
        ).to(self.llm.device)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: Optional[Tensor],
    ) -> Tensor:
        x = x.to(self.llm.device)
        edge_index = edge_index.to(self.llm.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.llm.device)
        batch = batch.to(self.llm.device)
        smiles = smiles.to(self.llm.device)

        out = self.gnn(x, edge_index, edge_attr=edge_attr)
        graph_embedding = scatter(out, batch, dim=0, reduce='mean')
        smiles_embedding = self.smiles_encoder(smiles)
        return graph_embedding, smiles_embedding

    def forward(self):
        pass

    @torch.no_grad()
    def inference(self):
        pass

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  llm={self.llm},\n'
                f'  gnn={self.gnn},\n'
                f')')
