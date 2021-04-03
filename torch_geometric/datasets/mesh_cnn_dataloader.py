import torch
from os import mkdir
import os.path as osp
from torch_geometric.data.dataloader import DataLoader
from torch_geometric.datasets.mesh_cnn_base_datasets import MeshCnnBaseDataset
from torch_geometric.transforms.mesh import Mesh


class MeshDataLoader:
    r"""This class creates a dataloader for MeshCNN datasets: `"MeshCNN: A
    Network with an Edge" <https://arxiv.org/abs/1809.05910>`_ paper

    Args:
        mesh_dataset (MeshCnnBaseDataset): mesh dataset with type of
                                           MeshCnnBaseDataset
                                           (MeshCnnSegmentationsDataset or
                                           MeshCnnClassificationDataset)
        data_set_type (str): dataset type according to task: 'classification'
                             or 'segmenations'.
        gpu_ids (int, list, optional): GPU IDs to use in the dataloader.
                                       Default is an empty list means CPU
                                       option.
        phase (str, optional): Train or Test phase - if 'train' it will return
                               the train datalodaer, if 'test' it will
                               return the test datalodaer. Default is 'train'.
        batch_size (int, optional): batch size for dataloader. Default is 1.
        shuffle (bool, optional): If `True` the dataloader will load data in a
                                  random manner. If `False` it will load the
                                  data in the same order. Default is `True`.
        hold_history (bool, optional): If `True` the Mesh structure will hold
                                       Mesh history. This property is used for
                                       MeshPool and MeshUnpool operations (see
                                       definitions).
        export_folder (str, optional): an export folder to create intermediate
                                       results for visualization. Default is an
                                       empty path which means no data export.
    """
    def __init__(self, mesh_dataset: MeshCnnBaseDataset, data_set_type: str,
                 gpu_ids=[], phase: str = 'train', batch_size: int = 1,
                 shuffle: bool = True, hold_history: bool = False,
                 export_folder: str = ''):

        self.dataset = mesh_dataset.dataset
        self.hold_history = hold_history
        self.data_set_type = data_set_type
        self.export_folder = export_folder
        if not osp.exists(export_folder) and not export_folder == '':
            mkdir(export_folder)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = self.dataset.n_classes
        self.n_input_channels = self.dataset.n_input_channels
        self.device = torch.device(
            'cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle)
        self.is_train = False
        if phase == 'train':
            self.is_train = True

    def __len__(self):
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        for j, data in enumerate(self.data_loader):
            if j * self.batch_size >= len(self.dataset):
                break
            else:
                meshes = []
                labels = []
                soft_label = []
                input_edge_features = []
                if self.data_set_type == 'classification':
                    labels = -1 * torch.ones(data.num_graphs).long()
                for idx in range(0, data.num_graphs):
                    d = data.get_example(idx)
                    meshes.append(
                        Mesh(mesh_data=d, hold_history=self.hold_history,
                             export_folder=self.export_folder))
                    if self.data_set_type == 'classification':
                        labels[idx] = d.label.item()
                    if self.data_set_type == 'segmentation':
                        labels.append(d.label)
                        soft_label.append(d.soft_label)
                    input_edge_features.append(d.features)

                input_edge_features = torch.tensor(input_edge_features).float()
                labels = torch.as_tensor(labels).long()
                # set inputs
                input_edge_features = input_edge_features.to(
                    self.device).requires_grad_(self.is_train)
                labels = labels.to(self.device)
                soft_label = torch.as_tensor(soft_label).to(self.device)
            yield meshes, input_edge_features, labels, soft_label
