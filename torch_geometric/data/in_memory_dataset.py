from torch_geometric.data import Dataset, Batch


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into memory.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
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
    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return []

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return []

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(self, root=None, transform=None, pre_transform=None,
                 pre_filter=None):
        super(InMemoryDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.__data__ = None

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        y = self.__data__.y
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    def len(self):
        return self.__data__.num_graphs

    def get(self, idx):
        return self.__data__[idx]

    def collate(self, data_list):
        r"""Collates a list of data objects to an internal storage format
        of :class:`torch_geometric.data.InMemoryDataset` for faster disk i/o.
        """
        self.__data__ = Batch.from_data_list(data_list)
        return self.__data__
