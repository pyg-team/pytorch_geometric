import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply


class CoMA(InMemoryDataset):
    r"""The CoMA 3D faces dataset from the `"Generating 3D faces using
    Convolutional Mesh Autoencoders" <https://arxiv.org/abs/1807.10267>`_
    paper, containing 20,466 meshes of extreme expressions captured over 12
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
        split_type (string, optional): The split type to use (:obj:`"sliced"`,
        :obj:`"expression"` or :obj:`"identity"`). (default: :obj:`"sliced"`)
        split_term (string, optional): If :attr:`split_type` is set to
            :obj:`"expression"` or :obj:`"identity"`, will use the
            :attr:`split_term` as the test set. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self, root, train=True, split_type='sliced', split_term=None,
                 transform=None, pre_transform=None):
        super(CoMA, self).__init__(root, transform, pre_transform)

        # TODO: We load the complete data here, despite only making use of
        # either the training or test set.
        self.data, self.slices = torch.load(self.processed_paths[0])
        indices = torch.load(self.processed_paths[1])

        assert split_type in ['sliced', 'expression', 'identity']

        if split_type == 'sliced':
            ind = torch.arange(len(self))
            mask = (ind % 100) >= 10 if train else (ind % 100) < 10
            self.__indices__ = ind[mask].tolist()

        elif split_type == 'expression':
            assert split_term in indices['expression']
            if train:
                ind = []
                for key, item in indices['expression'].items():
                    ind += item if key != split_term else []
                self.__indices__ = ind
            else:
                self.__indices__ = indices['expression'][split_term]

        elif split_type == 'identity':
            assert split_term in indices['identity']
            if train:
                ind = []
                for key, item in indices['identity'].items():
                    ind += item if key != split_term else []
                self.__indices__ = ind
            else:
                self.__indices__ = indices['identity'][split_term]

    @property
    def raw_file_names(self):
        return 'COMA_data.zip'

    @property
    def processed_file_names(self):
        return 'data.pt', 'indices.pt'

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download COMA_data.zip from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        paths = sorted(glob.glob(osp.join(self.raw_dir, '*/*/*.ply')))
        if len(paths) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            paths = sorted(glob.glob(osp.join(self.raw_dir, '*/*/*.ply')))

        indices = {'expression': {}, 'identity': {}}
        for expression in self.categories:
            indices['expression'][expression] = []
        for identity in glob.glob(osp.join(self.raw_dir, '*')):
            indices['identity'][identity.split(osp.sep)[-1]] = []

        data_list = []
        for idx, path in enumerate(paths):
            data = read_ply(path)
            data_list.append(data)

            indices['expression'][path.split(osp.sep)[-2]].append(idx)
            indices['identity'][path.split(osp.sep)[-3]].append(idx)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(indices, self.processed_paths[1])
