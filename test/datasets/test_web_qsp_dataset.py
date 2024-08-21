import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.datasets import UpdatedWebQSPDataset, WebQSPDataset
from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_web_qsp_dataset():
    dataset = WebQSPDataset()
    assert len(dataset) == 4700
    assert str(dataset) == "WebQSPDataset(4700)"


@onlyOnline
@onlyFullTest
def test_updated_web_qsp_dataset():
    def results_are_close_enough(ground_truth: Data, new_method: Data,
                                 thresh=.8):
        def _sorted_tensors_are_close(tensor1, tensor2):
            return torch.all(
                torch.isclose(tensor1.sort(dim=0)[0],
                              tensor2.sort(dim=0)[0]).float().mean(
                                  axis=1) > thresh)

        def _graphs_are_same(tensor1, tensor2):
            return nx.weisfeiler_lehman_graph_hash(nx.Graph(
                tensor1.T)) == nx.weisfeiler_lehman_graph_hash(
                    nx.Graph(tensor2.T))
        return _sorted_tensors_are_close(ground_truth.x, new_method.x) \
            and _sorted_tensors_are_close(ground_truth.edge_attr,
                                          new_method.edge_attr) \
            and _graphs_are_same(ground_truth.edge_index,
                                 new_method.edge_index)

    dataset_original = WebQSPDataset()
    dataset_updated = UpdatedWebQSPDataset('updated')

    for ground_truth, updated_graph in zip(dataset_original, dataset_updated):
        assert results_are_close_enough(ground_truth, updated_graph)
