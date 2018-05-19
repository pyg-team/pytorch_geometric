from torch_geometric.data import Dataset, data_from_set


class InMemoryDataset(Dataset):
    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __init__(self, root, transform=None):
        super(InMemoryDataset, self).__init__(root, transform)

        self.dataset = None
        self.slices = None

    def __len__(self):
        # TODO: change
        return self.slices[self.dataset.keys()[0]].size(0) - 1

    def get_data(self, i):
        # TODO: change
        return data_from_set(self.dataset, self.slices, i)
