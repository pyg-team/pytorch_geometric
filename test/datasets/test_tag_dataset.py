from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.datasets import TAGDataset
from torch_geometric.testing import withPackage


# @onlyFullTest
@withPackage('ogb')
def test_tag_dataset() -> None:
    root = './data'
    hf_model = 'prajjwal1/bert-tiny'
    token_on_disk = True

    dataset = PygNodePropPredDataset('ogbn-arxiv', root=root)
    tag_dataset = TAGDataset(root, dataset, hf_model,
                             token_on_disk=token_on_disk)

    assert 169343 == tag_dataset[0].num_nodes \
        == len(tag_dataset.text) \
        == len(tag_dataset.llm_explanation) \
        == len(tag_dataset.llm_prediction)
    assert 1166243 == tag_dataset[0].num_edges
