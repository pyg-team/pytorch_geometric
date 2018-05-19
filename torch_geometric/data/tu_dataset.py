from torch_geometric.data import InMemoryDataset


class TUDataset(InMemoryDataset):
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets/{}.zip'

    def __init__(self, root, name, transform=None):
        super(InMemoryDataset).__init__(root, transform)

        self.name = name

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        # url = self.url.format(self.url)
        pass

    def process(self):
        pass
