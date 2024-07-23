import torch
import torch.nn.functional as F
from torch import Tensor


def test_dropout():
    class MyModule(torch.nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return F.dropout(x, p=1.0, training=self.training)

    module = MyModule()
    graph_module = torch.fx.symbolic_trace(module)
    graph_module.recompile()

    x = torch.randn(4)

    graph_module.train()
    assert torch.allclose(graph_module(x), torch.zeros_like(x))

    # This is certainly undesired behavior due to tracing :(
    graph_module.eval()
    assert torch.allclose(graph_module(x), torch.zeros_like(x))
