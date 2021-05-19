from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.to_hetero import transform
from torch.fx import symbolic_trace


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 16)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = ModuleDict()
        self.lin1['paper'] = Linear(16, 16)
        self.lin1['author'] = Linear(16, 16)
        self.lin2 = ModuleDict()
        self.lin2['paper'] = Linear(16, 16)
        self.lin2['author'] = Linear(16, 16)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for key, item in x.items():
            x[key] = self.lin1[key](x[key]).relu_()
            x[key] = self.lin2[key](x[key]).relu_()
        return x


model1 = Net1()
model2 = Net2()

data = Planetoid('/tmp/Planetoid', 'Cora')[0]
data.x = data.x[:, :16]

metadata = (['paper', 'author'], [('paper', 'paper'), ('author', 'paper'),
                                  ('paper', 'author')])

gm = symbolic_trace(model2,
                    concrete_args={'x': {
                        'paper': data.x,
                        'author': data.x
                    }})

print(gm.graph.python_code('self'))
gm.graph.print_tabular()
print('-------------')
out_model = transform(model1, metadata, input_map={'x': 'node'})
out_model({'author': torch.randn(100, 16), 'paper': torch.randn(100, 16)})
