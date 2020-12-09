import csv
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data


class WordNet18RR(InMemoryDataset):
    r"""
    WordNet18 suffer from test leakage (Dettmers et al. 2018). They
    introduced the WN18RR.
    """

    edge_to_id = {
        "_also_see": 1,
        "_derivationally_related_form": 2,
        "_has_part": 3,
        "_hypernym": 4,
        "_instance_hypernym": 5,
        "_member_meronym": 6,
        "_member_of_domain_region": 7,
        "_member_of_domain_usage": 8,
        "_similar_to": 9,
        "_synset_domain_topic_of": 10,
        "_verb_group": 11,
    }

    urls = {
        "train": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/train.txt",
        "val": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/valid.txt",
        "test": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/test.txt",
    }

    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        download_url(self.urls["train"], self.raw_dir)
        download_url(self.urls["test"], self.raw_dir)
        download_url(self.urls["val"], self.raw_dir)

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def num_classes(self):
        return len(WordNet18RR.edge_to_id)

    def process(self):
        edge_index, edge_type = [], []
        n_train = 0
        n_val = 0
        n_test = 0
        for index, path in enumerate(self.raw_paths):
            with open(path) as csvfile:
                raw = list(csv.reader(csvfile, delimiter="\t"))
            edge_type.extend([WordNet18RR.edge_to_id[_[1]] for _ in raw])
            edge_index.extend([[int(_[0]), int(_[2])] for _ in raw])
            if 'train' in path:
                n_train = len(raw)
            elif 'val' in path:
                n_val = len(raw)
            else:
                n_test = len(raw)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        train_mask = [True]*n_train + [False]*(n_val+n_test)
        val_mask = [False]*n_train + [True]*n_val + [False]*n_test
        test_mask = [False]*(n_train+n_val) + [True]*n_test
        data_list = [Data(edge_index=edge_index, y=edge_type, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)]

        unique_vertices = edge_index.flatten().unique().tolist()
        self.vertice_to_id = dict(zip(unique_vertices, range(len(unique_vertices))))
        collated_data, slices = self.collate(data_list)
        torch.save((collated_data, slices), *self.processed_paths)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __len__(self):
        return 86835 + 3034 + 3134   # train + val + test

