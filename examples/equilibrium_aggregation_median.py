r"""
Replicates the experiment from `"Deep Graph Infomax"
<https://arxiv.org/abs/1809.10341>`_ to try and teach
`EquilibriumAggregation` to learn to take the median of
a set of numbers
"""

import torch

from torch_geometric.nn import EquilibriumAggregation

input_size = 1000
epochs = 100

model = EquilibriumAggregation(1, 1, [256, 256], 5, 0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

for i in range(epochs):
    optimizer.zero_grad()
    x = torch.rand(input_size, 1)
    y = model(x)
    loss = (y - x.median()).norm(2)
    print(y, x.median())
    loss.backward()
    optimizer.step()
    if i % 10 == 9:
        print(f"Loss at epoc {i} is {loss}")
