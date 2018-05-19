from torch_geometric.data import InMemoryDataset, download_url


class TUDataset(InMemoryDataset):
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets/'

    def __init__(self, root, name, transform=None):
        self.name = name
        super(TUDataset, self).__init__(root, transform)

    @property
    def raw_file_names(self):
        return 'awdawd'

    @property
    def processed_file_names(self):
        return 'awdawd'

    def download(self):
        download_url('{}/{}.zip'.format(self.url, self.name), self.raw_dir)
        pass

    def process(self):
        pass
