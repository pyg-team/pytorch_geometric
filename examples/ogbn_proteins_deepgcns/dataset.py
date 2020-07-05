from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import os.path
import torch_geometric as tg
import torch
import pickle
import scipy.sparse as sp
from torch_scatter import scatter
import random


class OGBNDataset(object):

    def __init__(self, dataset_name='ogbn-proteins'):
        """
        download the corresponding dataset based on the input name of
        dataset appointed the dataset will be divided into training,
        validation and test dataset the graph object will be obtained,
        which has three attributes
            edge_attr=[79122504, 8]
            edge_index=[2, 79122504]
            x=[132534, 8]
            y=[132534, 112]
        :param dataset_name:
        """
        self.dataset_name = dataset_name

        self.dataset = PygNodePropPredDataset(self.dataset_name)
        self.splitted_idx = self.dataset.get_idx_split()
        self.whole_graph = self.dataset[0]
        self.length = 1

        self.train_idx, self.valid_idx, self.test_idx = \
            self.splitted_idx["train"], self.splitted_idx["valid"], \
            self.splitted_idx["test"]
        self.num_tasks = self.dataset.num_tasks
        self.total_no_of_edges = self.whole_graph.edge_attr.shape[0]
        self.total_no_of_nodes = self.whole_graph.y.shape[0]
        self.species = self.whole_graph.node_species
        self.y = self.whole_graph.y

        self.edge_index = self.whole_graph.edge_index
        self.edge_attr = self.whole_graph.edge_attr

        self.x = self.generate_one_hot_encoding()
        # transpose and then convert it to numpy array type
        self.edge_index_array = self.edge_index.t().numpy()
        # obtain edge index dict
        self.edge_index_dict = self.edge_features_index()
        # obtain adjacent matrix
        self.adj = self.construct_adj()

    def generate_one_hot_encoding(self):

        le = preprocessing.LabelEncoder()
        species_unique = torch.unique(self.species)
        max_no = species_unique.max()
        le.fit(species_unique % max_no)
        species = le.transform(self.species.squeeze() % max_no)
        species = np.expand_dims(species, axis=1)

        enc = preprocessing.OneHotEncoder()
        enc.fit(species)
        one_hot_encoding = enc.transform(species).toarray()

        return torch.FloatTensor(one_hot_encoding)

    def extract_node_features(self, aggr='add'):

        file_path = 'init_node_features_{}.pt'.format(aggr)

        if os.path.isfile(file_path):
            print('{} exists'.format(file_path))
        else:
            if aggr in ['add', 'mean', 'max']:
                node_features = scatter(self.edge_attr,
                                        self.edge_index[0],
                                        dim=0,
                                        dim_size=self.total_no_of_nodes,
                                        reduce=aggr)
            else:
                raise Exception('Unknown Aggr Method')
            torch.save(node_features, file_path)
            print('Node features extracted are'
                  'saved into file {}'.format(file_path))
        return file_path

    def construct_adj(self):
        adj = sp.csr_matrix((np.ones(self.total_no_of_edges, dtype=np.uint8),
                             (self.edge_index_array[:, 0],
                              self.edge_index_array[:, 1])),
                            shape=(self.total_no_of_nodes,
                                   self.total_no_of_nodes))
        return adj

    def edge_features_index(self):
        file_name = 'edge_features_index_v2.pkl'
        if os.path.isfile(file_name):
            print('{} exists'.format(file_name))
            with open(file_name, 'rb') as edge_features_index:
                edge_index_dict = pickle.load(edge_features_index)
        else:
            df = pd.DataFrame()
            df['1st_index'] = self.whole_graph.edge_index[0]
            df['2nd_index'] = self.whole_graph.edge_index[1]
            df_reset = df.reset_index()
            key = zip(df_reset['1st_index'], df_reset['2nd_index'])
            edge_index_dict = df_reset.set_index(key)['index'].to_dict()
            with open(file_name, 'wb') as edge_features_index:
                pickle.dump(edge_index_dict, edge_features_index)
            print('Edges indexes information'
                  'is saved into file {}'.format(file_name))
        return edge_index_dict

    @staticmethod
    def random_partition_graph(num_nodes, cluster_number=100):
        parts = np.random.randint(cluster_number, size=num_nodes)
        return parts

    def generate_sub_graphs(self, parts, cluster_number=10, batch_size=1):
        no_of_batches = cluster_number // batch_size

        print('The number of clusters: {}'.format(cluster_number))

        sg_nodes = [[] for _ in range(no_of_batches)]
        sg_edges = [[] for _ in range(no_of_batches)]
        sg_edges_orig = [[] for _ in range(no_of_batches)]
        sg_edges_index = [[] for _ in range(no_of_batches)]

        edges_no = 0

        for cluster in range(no_of_batches):
            sg_nodes[cluster] = np.where(parts == cluster)[0]
            sg_edges[cluster] = tg.utils.from_scipy_sparse_matrix(
              self.adj[sg_nodes[cluster], :][:, sg_nodes[cluster]])[0]
            edges_no += sg_edges[cluster].shape[1]
            # mapper
            mapper = {nd_idx: nd_orig_idx
                      for nd_idx, nd_orig_idx in enumerate(sg_nodes[cluster])}
            # map edges to original edges
            sg_edges_orig[cluster] = \
                OGBNDataset.edge_list_mapper(mapper, sg_edges[cluster])
            # edge index
            sg_edges_index[cluster] = [self.edge_index_dict[(edge[0], edge[1])]
                                       for edge in
                                       sg_edges_orig[cluster].t().numpy()]
        print('Total number edges of sub graphs: {},'
              'of whole graph: {}, {:.2f} % edges are lost'.
              format(edges_no, self.total_no_of_edges,
                     (1 - edges_no / self.total_no_of_edges) * 100))

        return sg_nodes, sg_edges, sg_edges_index, sg_edges_orig

    @staticmethod
    def edge_list_mapper(mapper, sg_edges_list):
        idx_1st = list(map(lambda x: mapper[x], sg_edges_list[0].tolist()))
        idx_2nd = list(map(lambda x: mapper[x], sg_edges_list[1].tolist()))
        sg_edges_orig = torch.LongTensor([idx_1st, idx_2nd])
        return sg_edges_orig
