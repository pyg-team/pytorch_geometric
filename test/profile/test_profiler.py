from torch_geometric.profile.profiler import Profiler
import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE
from torch_geometric.testing import onlyFullTest, withCUDA


@onlyFullTest
@withCUDA
def test_torch_profiler(get_dataset, device):
    dataset = get_dataset(name='Cora')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)
    model.train()
    with Profiler(model, profile_memory=True, use_cuda=use_cuda) as prof:
        model(data.x, data.edge_index)
        print(prof)
