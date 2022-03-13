from torch_geometric.datasets.mesh_cnn_base_datasets import (
    MeshCnnClassificationDataset, MeshCnnSegmentationDataset)

from torch_geometric.transforms.mesh_prepare import MeshCNNPrepare


class MeshShrech16Dataset:
    r"""This class creates the classification shrech16 dataset for MeshCNN
    datasets: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper
    Args:
        root (str): Root folder for dataset.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train datalodaer, if 'False' it will return
                               the test datalodaer. Default is 'True'.
        num_aug (int, optional): Number of augmentations to apply on mesh.
                                 Default is 1 which means no augmentations.
        slide_verts (float, optional): values between 0.0 to 1.0 - if set above
                                       0 will apply shift along mesh surface
                                       with percentage of slide_verts value.
        scale_verts (bool, optional): If set to `False` - do not apply scale
                                      vertex augmentation. If set to `True` -
                                      apply non-uniformly scale on the mesh
        flip_edges (float, optional): values between 0.0 to 1.0 - if set above
                                      0 will apply random flip of edges with
                                      percentage of flip_edges value.
    """

    def __init__(self, root: str, train: bool = True, num_aug: int = 1,
                 slide_verts: float = 0.0,
                 scale_verts: bool = False, flip_edges: float = 0.0):
        self.root = root
        self.train = train
        self.num_aug = num_aug
        self.slide_verts = slide_verts
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.n_input_edges = 750
        self.data_set_type = 'classification'

        self.mesh_prepare = MeshCNNPrepare(aug_slide_verts=slide_verts,
                                           aug_scale_verts=scale_verts,
                                           aug_flip_edges=flip_edges,
                                           num_aug=num_aug)

        self.dataset = \
            MeshCnnClassificationDataset(root=self.root,
                                         name='shrec16',
                                         n_input_edges=self.n_input_edges,
                                         train=self.train,
                                         transform=self.mesh_prepare)

    def __len__(self):
        return len(self.dataset)


class MeshCubesDataset:
    r"""This class creates the classification cubes dataset for MeshCNN
    datasets: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper
    Args:
        root (str): Root folder for dataset.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train datalodaer, if 'False' it will return
                               the test datalodaer. Default is 'True'.
        num_aug (int, optional): Number of augmentations to apply on mesh.
                                 Default is 1 which means no augmentations.
        slide_verts (float, optional): values between 0.0 to 1.0 - if set above
                                       0 will apply shift along mesh surface
                                       with percentage of slide_verts value.
        scale_verts (bool, optional): If set to `False` - do not apply scale
                                      vertex augmentation. If set to `True` -
                                      apply non-uniformly scale on the mesh
        flip_edges (float, optional): values between 0.0 to 1.0 - if set above
                                      0 will apply random flip of edges with
                                      percentage of flip_edges value.
    """

    def __init__(self, root: str, train: bool = True, num_aug: int = 1,
                 slide_verts: float = 0.0,
                 scale_verts: bool = False, flip_edges: float = 0.0):
        self.root = root
        self.train = train
        self.num_aug = num_aug
        self.slide_verts = slide_verts
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges

        self.n_input_edges = 750
        self.data_set_type = 'classification'

        self.mesh_prepare = MeshCNNPrepare(aug_slide_verts=slide_verts,
                                           aug_scale_verts=scale_verts,
                                           aug_flip_edges=flip_edges,
                                           num_aug=num_aug)
        self.dataset = \
            MeshCnnClassificationDataset(root=self.root,
                                         name='cubes',
                                         n_input_edges=self.n_input_edges,
                                         train=self.train,
                                         transform=self.mesh_prepare)

    def __len__(self):
        return len(self.dataset)


class MeshHumanSegDataset:
    r"""This class creates the segmentation human_seg dataset for MeshCNN
    datasets: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper.
    Args:
        root (str): Root folder for dataset.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train datalodaer, if 'False' it will return
                               the test datalodaer. Default is 'True'.
        num_aug (int, optional): Number of augmentations to apply on mesh.
                                 Default is 1 which means no augmentations.
        slide_verts (float, optional): values between 0.0 to 1.0 - if set above
                                       0 will apply shift along mesh surface
                                       with percentage of slide_verts value.
        scale_verts (bool, optional): If set to `False` - do not apply scale
                                      vertex augmentation If set to `True` -
                                      apply non-uniformly scale on the mesh
        flip_edges (float, optional): values between 0.0 to 1.0 - if set above
                                      0 will apply random flip of edges with
                                      percentage of flip_edges value.
    """

    def __init__(self, root: str, train: bool = True, num_aug: int = 1,
                 slide_verts: float = 0.0,
                 scale_verts: bool = False, flip_edges: float = 0.0):
        self.root = root
        self.train = train
        self.num_aug = num_aug
        self.slide_verts = slide_verts
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.dataset_name = 'human_seg'
        self.n_input_edges = 2280
        self.data_set_type = 'segmentation'

        self.mesh_prepare = MeshCNNPrepare(aug_slide_verts=slide_verts,
                                           aug_scale_verts=scale_verts,
                                           aug_flip_edges=flip_edges,
                                           num_aug=num_aug)
        self.dataset = \
            MeshCnnSegmentationDataset(root=self.root,
                                       name=self.dataset_name,
                                       n_input_edges=self.n_input_edges,
                                       train=self.train,
                                       transform=self.mesh_prepare)

    def __len__(self):
        return len(self.dataset)


class MeshCoSegDataset:
    r"""This class creates the segmentation co_seg dataset for MeshCNN
    datasets: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper.
    Args:
        root (str): Root folder for dataset.
        sub_dataset (str, optional): sub-dataset in co_seg. Options are:
                                     coseg_aliens, coseg_chairs, coseg_vases.
                                     Default is coseg_aliens.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train datalodaer, if 'False' it will return
                               the test datalodaer. Default is 'True'.
        num_aug (int, optional): Number of augmentations to apply on mesh.
                                 Default is 1 which means no augmentations.
        slide_verts (float, optional): values between 0.0 to 1.0 - if set above
                                       0 will apply shift along mesh surface
                                       with percentage of slide_verts value.
        scale_verts (bool, optional): If set to `False` - do not apply scale
                                      vertex augmentation. If set to `True` -
                                      apply non-uniformly scale on the mesh
        flip_edges (float, optional): values between 0.0 to 1.0 - if set above
                                      0 will apply random flip of edges with
                                      percentage of flip_edges value.
    """

    def __init__(self, root: str, sub_dataset: str = 'coseg_aliens',
                 train: bool = True, num_aug: int = 1,
                 slide_verts: float = 0.0, scale_verts: bool = False,
                 flip_edges: float = 0.0):
        self.root = root
        self.train = train
        self.num_aug = num_aug
        self.slide_verts = slide_verts
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.dataset_name = sub_dataset
        self.n_input_edges = 2280
        self.data_set_type = 'segmentation'

        self.mesh_prepare = MeshCNNPrepare(aug_slide_verts=slide_verts,
                                           aug_scale_verts=scale_verts,
                                           aug_flip_edges=flip_edges,
                                           num_aug=num_aug)

        self.dataset = \
            MeshCnnSegmentationDataset(root=self.root,
                                       name=self.dataset_name,
                                       n_input_edges=self.n_input_edges,
                                       train=self.train,
                                       transform=self.mesh_prepare)

    def __len__(self):
        return len(self.dataset)
