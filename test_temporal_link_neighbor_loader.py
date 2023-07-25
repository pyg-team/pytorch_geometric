import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader


data = Data(
    x=torch.randn(10, 5),
    edge_index=torch.randint(0, 10, (2, 123)),
    edge_time=torch.arange(123, dtype=torch.long),
    edge_label=torch.ones(123),
)


loader = LinkNeighborLoader(
    data,
    num_neighbors=[2],
    edge_label=torch.ones(data.num_edges),
    time_attr="edge_time",
    edge_label_time=data.edge_time,
    batch_size=5,
    shuffle=True,
    disjoint=False,
)

for sample in loader:
    edge_index = sample.edge_index
    edge_time = sample.edge_time

    edge_label_index = sample.edge_label_index
    edge_label_time = sample.edge_label_time
    edge_label = sample.edge_label

    print("===input===")
    print(edge_index)
    print(edge_time)

    print("===target===")
    print(edge_label_index)
    print(edge_label_time)
    print(edge_label)

    print("===asdf===")
    # src/dst share the same timestamp
    edge_label_time2 = edge_label_time.repeat(2)
    print(edge_time - edge_label_time2[edge_index[1]])
    breakpoint()
    break
