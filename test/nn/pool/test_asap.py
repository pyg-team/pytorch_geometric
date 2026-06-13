import torch

from torch_geometric.testing import onlyLinux


@onlyLinux  # TODO  (matthias) Investigate CSR @ CSR support on Windows.
def test_asap():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    torch.randn((num_nodes, in_channels))
