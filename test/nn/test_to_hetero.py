import torch
from torch import Tensor
from torch.nn import Linear, ReLU
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, Sequential

model = Sequential('x, edge_index', [
    (GCNConv(16, 16), 'x , edge_index -> x'),
    ReLU(),
    (GCNConv(16, 16), 'x , edge_index -> x'),
    ReLU(),
])


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 16)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


model = Net()

data = Planetoid('/tmp/Planetoid', 'Cora')[0]
data.x = data.x[:, :16]

metadata = (['paper', 'author'], [('paper', 'paper'), ('author', 'paper'),
                                  ('paper', 'author')])

from torch.fx import symbolic_trace

gm = symbolic_trace(model)
print()
print(gm)
print()
gm.graph.print_tabular()
print()

for node in gm.graph.nodes:
    print(node)
    if node.op == 'call_module':
        print(getattr(gm, node.target))
