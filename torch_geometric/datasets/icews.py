import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.read import read_txt_array


class TemporalDataset(InMemoryDataset):
    url = 'https://github.com/INK-USC/RENet/raw/master/data'
    names = ['ICEWS18', 'GDELT']

    def __init__(self,
                 root,
                 name,
                 granularity,
                 transform=None,
                 pre_transform=None):
        assert name in self.names
        self.name = name
        self.granularity = granularity
        super(TemporalDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.splits = torch.load(self.processed_paths[1])

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def num_rels(self):
        return self.data.edge_type.max().item() + 1

    @property
    def raw_file_names(self):
        return ['{}.txt'.format(name) for name in ['train', 'valid', 'test']]

    @property
    def processed_file_names(self):
        return ['data.pt', 'splits.pt']

    def download(self):
        for filename in self.raw_file_names:
            url = '{}/{}/{}'.format(self.url, self.name, filename)
            download_url(url, self.raw_dir)

    def process(self):
        data_list = []
        splits = [0]
        for raw_path in self.raw_paths:
            srot = read_txt_array(raw_path, sep='\t', end=4, dtype=torch.long)
            row, rel, col, time = srot.t().contiguous()
            time = time / self.granularity

            count = time.bincount()
            split_sections = count[count > 0].tolist()

            rows = row.split(split_sections)
            cols = col.split(split_sections)
            rels = rel.split(split_sections)
            times = time.split(split_sections)
            splits.append(splits[-1] + len(rows))

            for row, col, rel, time in zip(rows, cols, rels, times):
                edge_index = torch.stack([row, col], dim=0)
                data = Data(edge_index=edge_index, edge_type=rel, time=time)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(splits, self.processed_paths[1])


class ICEWS(TemporalDataset):
    r"""The Integrated Crisis Early Warning System (ICEWS) dataset used in
    the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 1/1/2014 to 12/31/2014 (24 hours time granularity).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(ICEWS, self).__init__(root, 'ICEWS18', 24, transform,
                                    pre_transform)
