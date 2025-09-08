import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.llm.models import LLM, MoleculeGPT, SentenceTransformer
from torch_geometric.nn import GINEConv
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_molecule_gpt() -> None:
    llm = LLM(
        # model_name='lmsys/vicuna-7b-v1.5',
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        num_params=1,
        dtype=torch.bfloat16,
    )

    graph_encoder = GINEConv(nn=Seq(Lin(16, 16), ReLU(), Lin(16, 16)),
                             train_eps=True, edge_dim=16)

    smiles_encoder = SentenceTransformer(
        model_name='DeepChem/ChemBERTa-77M-MTR',
        pooling_strategy='last_hidden_state',
    )

    model = MoleculeGPT(
        llm=llm,
        graph_encoder=graph_encoder,
        smiles_encoder=smiles_encoder,
    )

    assert str(model) == (
        'MoleculeGPT(\n'
        '  llm=LLM(TinyLlama/TinyLlama-1.1B-Chat-v0.1),\n'
        '  graph=GINEConv,\n'
        '  smiles=SentenceTransformer(model_name=DeepChem/ChemBERTa-77M-MTR),\n'  # noqa: E501
        ')')

    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = torch.randn(edge_index.size(1), 16)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    smiles = ['CCCCCCCCCC']
    instructions = ['What is ∼ functional related to?']
    label = ['I do not know!']

    # Test train:
    loss = model(x, edge_index, batch, edge_attr, smiles, instructions, label)
    assert loss >= 0

    # Test inference:
    pred = model.inference(x, edge_index, batch, edge_attr, smiles,
                           instructions)
    assert len(pred) == 1
