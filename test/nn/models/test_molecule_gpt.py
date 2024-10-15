import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn import GINConv, MoleculeGPT
from torch_geometric.nn.nlp import LLM


def test_molecule_gpt() -> None:
    llm = LLM(
        model_name='lmsys/vicuna-7b-v1.5',
        num_params=7,
        dtype=torch.bfloat16,
    )

    graph_encoder = GINConv(nn=Seq(Lin(16, 32), ReLU(), Lin(32, 32)),
                            train_eps=True)

    smiles_encoder = LLM(
        model_name='DeepChem/ChemBERTa-77M-MTR',
        num_params=1,
        dtype=torch.bfloat16,
    )

    model = MoleculeGPT(
        llm=llm,
        graph_encoder=graph_encoder,
        smiles_encoder=smiles_encoder,
        use_lora=True,
        mlp_out_channels=4096,
    )

    assert 'MoleculeGPT' in str(model)
