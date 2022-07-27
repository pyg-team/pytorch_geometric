r"""
Replicates the experiment from `"Deep Graph Infomax"
<https://arxiv.org/abs/1809.10341>`_ to try and teach
`EquilibriumAggregation` to learn to take the median of
a set of numbers

This example converges slowly to being able to predict the
median similar to what is observed in the paper.
"""

import numpy as np
import torch

from torch_geometric.nn.aggr import EquilibriumAggregation

input_size = 100
steps = 10000000
embedding_size = 10
eval_each = 1000

model = EquilibriumAggregation(1, 10, [256, 256], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

norm = torch.distributions.normal.Normal(0.5, 0.4)
gamma = torch.distributions.gamma.Gamma(0.2, 0.5)
uniform = torch.distributions.uniform.Uniform(0, 1)
total_loss = 0
n_loss = 0

for i in range(steps):
    optimizer.zero_grad()
    dist = np.random.choice([norm, gamma, uniform])
    x = dist.sample((input_size, 1))
    y = model(x)
    loss = (y - x.median()).norm(2) / input_size
    loss.backward()
    optimizer.step()
    total_loss += loss
    n_loss += 1
    if i % eval_each == (eval_each - 1):
        print(f"Average loss at epoc {i} is {total_loss / n_loss}")
