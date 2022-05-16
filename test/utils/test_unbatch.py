import torch

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.utils import unbatch


def test_unbatch():
    # create test graphs
    G_1 = Data(x=Tensor([1,3,5]),edge_index=Tensor([[0,0,1],[1,2,2]]),edge_attr=Tensor([0,1,0]),num_nodes=3)
    G_2 = Data(x=Tensor([2,4,6,8,10]),edge_index=Tensor([[0,0,0,0,1,1,1,2,2,3],[1,2,3,4,2,3,4,3,4,4]]),edge_attr=Tensor([0,1,0,0,0,0,1,1,1,1]),num_nodes=5)
    G_3 = Data(x=Tensor([1,2]),edge_index=Tensor([[0],[1]]),edge_attr=Tensor([0]),num_nodes=2)
    x_test = [G_1.x,G_2.x,G_3.x]
    edge_attr_test = [G_1.edge_attr,G_2.edge_attr,G_3.edge_attr]
    # create and load dataset
    loader = DataLoader([G_1,G_2,G_3],batch_size=3,shuffle=False,follow_batch=['x','edge_attr'])
    for data in loader:
        x_unbatched = unbatch(data.x,data.x_batch,0)
        edge_attr_unbatched = unbatch(data.edge_attr,data.edge_attr_batch,0)
        # check if unbatched tensors match original
        for i in range(0,3):
            assert torch.equal(x_unbatched[i],x_test[i]) == True
            assert torch.equal(edge_attr_unbatched[i],edge_attr_test[i]) == True
