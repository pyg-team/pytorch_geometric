import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

data = Data(
    x=torch.randn(10, 5),
    edge_index=torch.randint(0, 10, (2, 1234)),
    edge_time=torch.arange(1234, dtype=torch.long),
    edge_label=torch.ones(1234),
)

loader = LinkNeighborLoader(
    data,
    num_neighbors=[10],
    edge_label=data.edge_label,
    time_attr="edge_time",
    edge_label_time=data.edge_time,
    batch_size=1,
    shuffle=True,
)

for sample in loader:
    print(sample)
    # print(sample.edge_time <= sample.edge_label_time)
    # assert torch.all(sample.edge_time <= sample.edge_label_time)
    print(sample.input_id)
    break


import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

data = Data(
    x=torch.randn(10, 5),
    node_time=torch.arange(10, dtype=torch.long),
    edge_index=torch.randint(0, 10, (2, 1234)),
    edge_label=torch.ones(1234),
)

loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    time_attr="node_time",
    input_time=data.node_time,
    batch_size=1,
    shuffle=True,
)

for sample in loader:
    print(sample)
    print(sample.input_id)
    # print(sample.node_time)
    # print(sample.seed_time)
    # assert torch.all(sample.node_time <= sample.node_label_time)
    break

"""
Data(x=[18, 5], edge_index=[2, 20], edge_time=[20], edge_label=[1],  n_id=[18], e_id=[20], batch=[18], num_sampled_nodes=[2], num_sampled_edges=[1], input_id=[1], edge_label_index=[2, 1], edge_label_time=[1])
Data(x=[9, 5],  edge_index=[2, 60], node_time=[9],  edge_label=[60], n_id=[9],  e_id=[60], batch=[9],  num_sampled_nodes=[3], num_sampled_edges=[2], input_id=[1], seed_time=[1], batch_size=1)
"""
