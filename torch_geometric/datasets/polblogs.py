import tarfile
import os

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected


class Polblogs(InMemoryDataset):
    """ The Political Blogs dataset (Polblogs) was introduced 
    by Adamic, L. A., & Glance, N. in "The political blogosphere 
    and the 2004 US election: divided they blog", in Proceedings of the 
    WWW-2005 Workshop on the Weblogging Ecosystem (2005). 

    Polblogs is a graph with 1,490 vertices (representing political
    blogs) and 19,025 edges (links between blogs). These links are
    automatically extracted from a crawl of the front page of the blog. 
    Each vertex receives a label indicating the political leaning of the blog: 
    liberal or conservative. Edges are undirected. 

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

    url = 'https://netset.telecom-paris.fr/datasets/polblogs.tar.gz'

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'polblogs.csv'

    @ property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # If already downloaded
        if os.path.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
           return
        # Download to `self.raw_dir`.
        path = download_url(self.url, self.raw_dir)
        path = os.path.abspath(path)
        with tarfile.open(path) as f:
            f.extractall(os.path.dirname(path))
        os.unlink(path)

    def process(self):
        dirname = os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), self.raw_dir)
        A = pd.read_csv(dirname + '/adjacency.csv',
                        header=None, index_col=False)
        A = torch.tensor(A.to_numpy().T)
        labels = pd.read_csv(dirname + '/labels.csv', header=None)
        labels = torch.tensor(labels.to_numpy().T[0])
        # one label was forgotten
        labels = torch.cat((torch.tensor([0]), labels), dim=0)

        # Construct PyG dataframe
        data = Data(edge_index=A, y=labels)
        data.edge_index = to_undirected(data.edge_index)
        data.num_classes = 2
        data.y = labels
        data.num_nodes = data.y.shape[0]
        data.x = torch.ones(data.num_nodes, 9)
        data.x = torch.cat((data.x, data.y.unsqueeze(1)), dim=1)

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])
