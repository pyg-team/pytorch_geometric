from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.to_hetero import transform
from torch.fx import symbolic_trace


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        # self.conv1 = SAGEConv(16, 16)
        self.lin2 = Linear(16, 16)

    def forward(self, x: Tensor, edge_index: Tensor, y: Tensor) -> Tensor:
        x = self.lin1(x).relu_()
        y = self.lin1(y).relu_()
        x = x + x
        return x, y


model = Net1()

metadata = (['paper', 'author'], [('paper', '_', 'paper'),
                                  ('author', '_', 'paper'),
                                  ('paper', '_', 'author'),
                                  ('author', '_', 'author')])

x_dict = {'author': torch.randn(100, 16), 'paper': torch.randn(100, 16)}
edge_index_dict = {
    ('paper', '_', 'paper'): torch.randint(100, (2, 200), dtype=torch.long),
    ('paper', '_', 'author'): torch.randint(100, (2, 200), dtype=torch.long),
    ('author', '_', 'paper'): torch.randint(100, (2, 200), dtype=torch.long),
    ('author', '_', 'author'): torch.randint(100, (2, 200), dtype=torch.long),
}
y_dict = {'author': torch.randn(100, 16), 'paper': torch.randn(100, 16)}

out_model = transform(model, metadata)
out_model(x_dict, edge_index_dict, y_dict)

# print(out_model.modules())

# class Net2(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = ModuleDict()
#         self.lin1['paper'] = Linear(16, 16)
#         self.lin1['author'] = Linear(16, 16)
#         self.lin2 = ModuleDict()
#         self.lin2['paper'] = Linear(16, 16)
#         self.lin2['author'] = Linear(16, 16)

#     def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         for key, item in x.items():
#             x[key] = self.lin1[key](x[key]).relu_()
#             x[key] = self.lin2[key](x[key]).relu_()
#         return x

# model2 = Net2()
# gm = symbolic_trace(model2,
#                     concrete_args={'x': {
#                         'paper': data.x,
#                         'author': data.x
#                     }})

# print(gm.graph.python_code('self'))
# gm.graph.print_tabular()
