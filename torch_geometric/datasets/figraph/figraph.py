import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx


class FiGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.custom_root = root
        super(FiGraphDataset, self).__init__(root, transform, pre_transform)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    @property
    def raw_dir(self):
        return os.path.join(self.custom_root, 'no_raw_folder')

    def process(self):
        data_list = []
        data_dir = os.path.join(self.custom_root, 'data')

        for year in range(2014, 2016):
            edge_path = os.path.join(data_dir, f'edges{year}.csv')
            edge_data = pd.read_csv(edge_path, header=None)  

            G = nx.Graph()
            for _, row in edge_data.iterrows():
                G.add_edge(row[0], row[1], edge_type=row[2])

            data = from_networkx(G)
            data.year = torch.tensor([year], dtype=torch.long)

            feature_path = os.path.join(data_dir, 'ListedCompanyFeatures.csv')
            feature_data = pd.read_csv(feature_path)

            node_features = []
            node_labels = []
            node_ids = []
            for _, row in feature_data[feature_data['Year'] == year].iterrows():
                node_id = row['nodeID']
                node_ids.append(node_id)

                try:
                    features = torch.tensor(row.drop(['nodeID', 'Year', 'Label']).values.astype(float),
                                            dtype=torch.float)
                except ValueError as e:
                    print(f"Error converting features for node {node_id} in year {year}: {e}")
                    continue

                node_features.append(features)
                label = torch.tensor([row['Label']], dtype=torch.long)
                node_labels.append(label)

            if len(node_features) > 0:
                data.x = torch.stack(node_features)
            if len(node_labels) > 0:
                data.y = torch.cat(node_labels)

            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        pass


if __name__ == '__main__':
    dataset = FiGraphDataset(root='./')
    print("Data processing complete. Processed data saved.")