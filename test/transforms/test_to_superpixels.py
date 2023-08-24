import os.path as osp

import torch

from torch_geometric.data import download_url, extract_gz
from torch_geometric.data.makedirs import makedirs
from torch_geometric.loader import DataLoader
from torch_geometric.testing import onlyOnline, withPackage
from torch_geometric.transforms import ToSLIC

resources = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
]


@onlyOnline
@withPackage('torchvision', 'skimage')
def test_to_superpixels(tmp_path):
    import torchvision.transforms as T
    from torchvision.datasets.mnist import (
        MNIST,
        read_image_file,
        read_label_file,
    )

    raw_folder = osp.join(tmp_path, 'MNIST', 'raw')
    processed_folder = osp.join(tmp_path, 'MNIST', 'processed')

    makedirs(raw_folder)
    makedirs(processed_folder)
    for resource in resources:
        path = download_url(resource, raw_folder)
        extract_gz(path, osp.join(tmp_path, raw_folder))

    test_set = (
        read_image_file(osp.join(raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(osp.join(raw_folder, 't10k-labels-idx1-ubyte')),
    )

    torch.save(test_set, osp.join(processed_folder, 'training.pt'))
    torch.save(test_set, osp.join(processed_folder, 'test.pt'))

    dataset = MNIST(tmp_path, download=False)

    dataset.transform = T.Compose([T.ToTensor(), ToSLIC()])

    data, y = dataset[0]
    assert len(data) == 2
    assert data.pos.dim() == 2 and data.pos.size(1) == 2
    assert data.x.dim() == 2 and data.x.size(1) == 1
    assert data.pos.size(0) == data.x.size(0)
    assert y == 7

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch, y in loader:
        assert batch.num_graphs == len(batch) == 2
        assert batch.pos.dim() == 2 and batch.pos.size(1) == 2
        assert batch.x.dim() == 2 and batch.x.size(1) == 1
        assert batch.batch.dim() == 1
        assert batch.ptr.dim() == 1
        assert batch.pos.size(0) == batch.x.size(0) == batch.batch.size(0)
        assert y.tolist() == [7, 2]
        break

    dataset.transform = T.Compose(
        [T.ToTensor(), ToSLIC(add_seg=True, add_img=True)])

    data, y = dataset[0]
    assert len(data) == 4
    assert data.pos.dim() == 2 and data.pos.size(1) == 2
    assert data.x.dim() == 2 and data.x.size(1) == 1
    assert data.pos.size(0) == data.x.size(0)
    assert data.seg.size() == (1, 28, 28)
    assert data.img.size() == (1, 1, 28, 28)
    assert data.seg.max().item() + 1 == data.x.size(0)
    assert y == 7

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch, y in loader:
        assert batch.num_graphs == len(batch) == 2
        assert batch.pos.dim() == 2 and batch.pos.size(1) == 2
        assert batch.x.dim() == 2 and batch.x.size(1) == 1
        assert batch.batch.dim() == 1
        assert batch.ptr.dim() == 1
        assert batch.pos.size(0) == batch.x.size(0) == batch.batch.size(0)
        assert batch.seg.size() == (2, 28, 28)
        assert batch.img.size() == (2, 1, 28, 28)
        assert y.tolist() == [7, 2]
        break
