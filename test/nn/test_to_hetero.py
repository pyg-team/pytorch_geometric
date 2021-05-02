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

# HOW TO ADD MODULES? LAZY MODULES


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if type(modules[node.target]) is nn.BatchNorm2d and type(
                modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model
