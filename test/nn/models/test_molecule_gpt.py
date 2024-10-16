import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn import GINEConv, MoleculeGPT
from torch_geometric.nn.nlp import LLM


def test_molecule_gpt() -> None:
    llm = LLM(
        # model_name='lmsys/vicuna-7b-v1.5',
        model_name='DeepChem/ChemBERTa-77M-MTR',
        num_params=7,
        dtype=torch.bfloat16,
    )

    graph_encoder = GINEConv(nn=Seq(Lin(16, 32), ReLU(), Lin(32, 32)),
                             train_eps=True, edge_dim=16)

    smiles_encoder = LLM(
        model_name='DeepChem/ChemBERTa-77M-MTR',
        num_params=1,
        dtype=torch.bfloat16,
    )

    model = MoleculeGPT(
        llm=llm,
        graph_encoder=graph_encoder,
        smiles_encoder=smiles_encoder,
        mlp_out_channels=4096,
    )

    assert 'MoleculeGPT' in str(model)

    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = torch.randn(edge_index.size(1), 16)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    smiles = ['CCCCCCCCCC']

    model(x, edge_index, batch, edge_attr, smiles)
