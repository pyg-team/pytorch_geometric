import glob
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply


class FAUST(InMemoryDataset):
    r"""The FAUST humans dataset from the `"FAUST: Dataset and Evaluation for
    3D Mesh Registration"
    <http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>`_ paper,
    containing 100 watertight meshes representing 10 different poses for 10
    different subjects.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'http://faust.is.tue.mpg.de/'

    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(FAUST, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'MPI-FAUST.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')

        paths = sorted(glob.glob(osp.join(path, '*.ply')))
        if len(paths) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            paths = sorted(glob.glob(osp.join(path, '*.ply')))

        data_list = [read_ply(path) for path in paths]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])
