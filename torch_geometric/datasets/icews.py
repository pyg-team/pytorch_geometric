import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.io import read_txt_array


class EventDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(EventDataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)

    @property
    def num_nodes(self):
        raise NotImplementedError

    @property
    def num_rels(self):
        raise NotImplementedError

    def process_events(self):
        raise NotImplementedError

    def process(self):
        events = self.process_events()
        events = events - events.min(dim=0, keepdim=True)[0]

        data_list = []
        for (sub, rel, obj, t) in events.tolist():
            data = Data(sub=sub, rel=rel, obj=obj, t=t)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list


class ICEWS18(EventDataset):
    r"""The Integrated Crisis Early Warning System (ICEWS) dataset used in
    the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity).

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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://github.com/INK-USC/RENet/raw/master/data/ICEWS18'
    splits = [0, 373018, 419013, 468558]  # Train/Val/Test splits.

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        assert split in ['train', 'val', 'test']
        super(ICEWS18, self).__init__(root, transform, pre_transform,
                                      pre_filter)
        idx = self.processed_file_names.index('{}.pt'.format(split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def num_nodes(self):
        return 23033

    @property
    def num_rels(self):
        return 256

    @property
    def raw_file_names(self):
        return ['{}.txt'.format(name) for name in ['train', 'valid', 'test']]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for filename in self.raw_file_names:
            download_url('{}/{}'.format(self.url, filename), self.raw_dir)

    def process_events(self):
        events = []
        for path in self.raw_paths:
            data = read_txt_array(path, sep='\t', end=4, dtype=torch.long)
            data[:, 3] = data[:, 3] / 24
            events += [data]
        return torch.cat(events, dim=0)

    def process(self):
        s = self.splits
        data_list = super(ICEWS18, self).process()
        torch.save(self.collate(data_list[s[0]:s[1]]), self.processed_paths[0])
        torch.save(self.collate(data_list[s[1]:s[2]]), self.processed_paths[1])
        torch.save(self.collate(data_list[s[2]:s[3]]), self.processed_paths[2])
