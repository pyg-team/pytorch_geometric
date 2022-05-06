r"""
Replicates the experiment from `"Deep Graph Infomax"
<https://arxiv.org/abs/1809.10341>`_ to try and teach
`EquilibriumAggregation` to learn to take the median of
a set of numbers
"""

import numpy as np
import torch

from torch_geometric.nn import EquilibriumAggregation

input_size = 100
epochs = 10000
embedding_size = 10

model = torch.nn.Sequential(EquilibriumAggregation(1, 10, [256, 256], 5, 0.1),
                            torch.nn.Linear(10, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

norm = torch.distributions.normal.Normal(0, 0.5)
gamma = torch.distributions.gamma.Gamma(1, 2)
uniform = torch.distributions.uniform.Uniform(-1, 1)

total_loss = 0
n_loss = 0
for i in range(epochs):
    optimizer.zero_grad()
    dist = np.random.choice([norm, gamma, uniform])
    x = dist.sample((input_size, 1))
    y = model(x)
    loss = (y - x.median()).norm(2)
    loss.backward()
    optimizer.step()
    total_loss += loss
    n_loss += 1
    if i % 500 == 499:
        print(f"Average loss at epoc {i} is {total_loss/n_loss}")
