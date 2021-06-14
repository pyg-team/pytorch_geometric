import os
import urllib.request
import torch
from torch_geometric.data import InMemoryDataset, Data


class LinkPrediction(InMemoryDataset):

    def __init__(self, root='../data', name='rdf_test', transform=None,
                 pre_transform=None):
        r"""The datasets FB15k237 for link prediction evaluation from
            the `"Modeling Relational Data with Graph Convolutional Networks"
            <https://arxiv.org/abs/1703.06103>`_ paper.
            Training and test splits are given by sets of triples. The data does contain default one-hot encoded node
            features.

            Args:
                root (string): Root directory where the dataset should be saved.
                name (string): The name of the dataset (:obj:`"FB15k-237"`).
                transform (callable, optional): A function/transform that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a transformed
                    version. The data object will be transformed before every access.
                    (default: :obj:`None`)
                pre_transform (callable, optional): A function/transform that takes in
                    an :obj:`torch_geometric.data.Data` object and returns a
                    transformed version. The data object will be transformed before
                    being saved to disk. (default: :obj:`None`)
        """

        assert name in ['FB15k-237']
        self.name = name
        super(LinkPrediction, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.num_relations = self.data.num_relations.tolist()

    def download(self):
        if self.name == 'FB15k-237':
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/README.txt", f"{self.raw_dir}/README.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/entities.dict", f"{self.raw_dir}/entities.dict")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/relations.dict", f"{self.raw_dir}/relations.dict")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/test.txt", f"{self.raw_dir}/test.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/train.txt", f"{self.raw_dir}/train.txt")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova/valid.txt", f"{self.raw_dir}/valid.txt")

    def process(self):
        x, edges_train_valid_test = read_graph_dicts(self.raw_dir)
        data = Data(x=x,
                    train_edge_index=torch.reshape(edges_train_valid_test['train']['edge_index'], (2, -1)), train_edge_type=edges_train_valid_test['train']['edge_type'],
                    valid_edge_index=torch.reshape(edges_train_valid_test['valid']['edge_index'], (2, -1)), valid_edge_type=edges_train_valid_test['valid']['edge_type'],
                    test_edge_index=torch.reshape(edges_train_valid_test['test']['edge_index'], (2, -1)), test_edge_type=edges_train_valid_test['test']['edge_type'])

        unique_nodes = torch.unique(torch.cat((data.train_edge_index, data.valid_edge_index, data.test_edge_index), 1))
        data.num_nodes = max(unique_nodes.size(0), torch.max(unique_nodes) + 1)
        data.num_relations = torch.unique(torch.cat((data.train_edge_type, data.valid_edge_type, data.test_edge_type), 0)).max().item() + 1
        data.x = torch.eye(data.num_nodes)

        self.data, self.slices = self.collate([data])
        torch.save((self.data, self.slices), self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return ['entities.dict', 'relations.dict', 'test.txt', 'train.txt', 'valid.txt']

    @property
    def num_relations(self):
        return self.data.num_relations

    @property
    def num_nodes(self):
        return self.data.num_nodes


def read_graph_dicts(directory):

    entities_dict = {}
    for line in open(os.path.join(directory, 'entities.dict'), 'r+'):
        line = line.strip().split('\t')
        entities_dict[line[1]] = int(line[0])

    relations_dict = {}
    for line in open(os.path.join(directory, 'relations.dict'), 'r+'):
        line = line.strip().split('\t')
        relations_dict[line[1]] = int(line[0])

    edges_train_valid_test = {}
    for edge_index_subset in ['train', 'valid', 'test']:
        subjects = []
        relations = []
        objects = []
        for line in open(os.path.join(directory, f'{edge_index_subset}.txt'), 'r+'):
            subject, relation, object = line.strip().split('\t')
            subjects.append(entities_dict[subject])
            relations.append(relations_dict[relation])
            objects.append(entities_dict[object])
        edges_train_valid_test[edge_index_subset] = {'edge_index': torch.tensor([subjects, objects]),
                                                     'edge_type': torch.tensor(relations)}

    x = torch.tensor([])

    return x, edges_train_valid_test





