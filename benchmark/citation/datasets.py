from pathlib import Path

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = Path.joinpath(Path(__file__).resolve().parent.parent,'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset
