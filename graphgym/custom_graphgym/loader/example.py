from torch_geometric.datasets import QM7b

from torch_geometric.graphgym.register import register_dataset

register_dataset('qm7b', QM7b)
