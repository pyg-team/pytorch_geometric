from typing import Dict, Tuple, List

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, LazyLinear, Module, ModuleDict
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn import MessagePassing

model = Sequential('x, edge_index', [
    (GCNConv(16, 16), 'x , edge_index -> x'),
    ReLU(),
    (GCNConv(16, 16), 'x , edge_index -> x'),
    ReLU(),
])


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = LazyLinear(16)
        self.lin2 = Linear(16, 16)
        self.conv1 = GCNConv(16, 16)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 16)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for key, item in x.items():
            x[key] = self.lin1(x[key]).relu_()
            x[key] = self.lin2(x[key]).relu_()
        return x


model1 = Net1()
model2 = Net2()

data = Planetoid('/tmp/Planetoid', 'Cora')[0]
data.x = data.x[:, :16]

metadata = (['paper', 'author'], [('paper', 'paper'), ('author', 'paper'),
                                  ('paper', 'author')])

from torch.fx import symbolic_trace

gm = symbolic_trace(model2,
                    concrete_args={'x': {
                        'paper': data.x,
                        'author': data.x
                    }})
gm = symbolic_trace(model1)
# print()
# print(gm)
# print()
# gm.graph.print_tabular()
# print()

Metagraph = Tuple[List[str], List[Tuple[str, str, str]]]


def duplicate_modules_(module: Module, metagraph: Metagraph):
    modules = dict(module.named_children())
    for key, item in modules.items():
        if isinstance(item, MessagePassing):
            pass
        else:
            pass


print(isinstance(model1.conv1, MessagePassing))
duplicate_modules_(model1, (['paper', 'author'], []))

# gm.graph.call_module(model1.lin)

# for node in gm.graph.nodes:
#     print(node)
#     if node.op == 'call_module':
#         print(getattr(gm, node.target))

# HOW TO ADD MODULES? LAZY MODULES

# def fuse(model: torch.nn.Module) -> torch.nn.Module:
#     model = copy.deepcopy(model)
#     fx_model: fx.GraphModule = fx.symbolic_trace(model)
#     modules = dict(fx_model.named_modules())

for node in list(gm.graph.nodes):
    if node.op != 'call_module':
        continue

    with gm.graph.inserting_after(node):
        new_node = gm.graph.call_module(node.target, args=(node, ))
        # node.replace_all_uses_with(new_node)

# print()
# gm.graph.print_tabular()
# gm.graph.lint()
# gm.recompile()
