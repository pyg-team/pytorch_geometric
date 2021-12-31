from torch_geometric.datasets import QM7b

from torch_geometric.graphgym.register import register_loader


@register_loader('qm7b')
def qm7b_dataset(root_dir: str):
    return QM7b(root_dir)
