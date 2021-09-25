import os
import os.path as osp
import numpy as np
import pickle
from torch_geometric.data import (download_url, extract_tar, InMemoryDataset)
from torch_geometric.transforms.mesh_prepare import MeshCNNPrepare
from torch_geometric.io import read_ply
import ntpath
from torch_geometric.data import Data


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant',
                  constant_values=val)


class MeshCnnBaseDataset(InMemoryDataset):
    r"""Base class for MeshCNN datasets: `"MeshCNN: A Network with an Edge"
     <https://arxiv.org/abs/1809.05910>`_ paper.
    This class is the base class for MeshCnnClassificationDataset and
    MeshCnnSegmentationDataset (see definitions).
    Args:
        root (str): Root folder for dataset.
        dataset_url (str): dataset URL link. Official supported URLs:
        dataset for classification:
        https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz?dl=1
        https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz?dl=1
        dataset(s) for segmentation:
        https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz?dl=1
        https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz?dl=1
    """

    def __init__(self, root: str, dataset_url: str):
        self.root = root
        self.dataset_url = dataset_url
        self.dir = None
        self.mean = 0
        self.std = 1
        self.n_input_channels = None
        self.size = None
        self.n_classes = None
        super(MeshCnnBaseDataset, self).__init__(self.root)

    @property
    def download(self):
        pass

    @property
    def raw_file_names(self):
        # A list of files in the raw_dir which needs to be found in order to
        # skip the download.
        return [osp.join(self.raw_dir,
                         osp.basename(self.dataset_url).replace('?dl=1', ''))]

    @property
    def processed_file_names(self):
        # A list of files in the process_dir which needs to be found in order
        # to skip the process.
        return ['abc']  # todo handle this

    @staticmethod
    def get_data_from_file(file_path):
        filename = ntpath.split(file_path)[1]
        fullfilename = file_path
        _, suffix = os.path.splitext(file_path)

        if suffix == '.ply':
            data = read_ply(file_path)
            pos = data.pos
            faces = np.array(data.face.transpose(0, 1), dtype=int)
        elif suffix == '.obj':
            pos, faces = [], []
            f = open(file_path)
            for line in f:
                line = line.strip()
                splitted_line = line.split()
                if not splitted_line:
                    continue
                elif splitted_line[0] == 'v':
                    pos.append([float(v) for v in splitted_line[1:4]])
                elif splitted_line[0] == 'f':
                    face_vertex_ids = [int(c.split('/')[0]) for c in
                                       splitted_line[1:]]
                    assert len(face_vertex_ids) == 3
                    face_vertex_ids = [
                        (ind - 1) if (ind >= 0) else (len(pos) + ind)
                        for ind in face_vertex_ids]
                    faces.append(face_vertex_ids)
            f.close()
            faces = np.array(faces, dtype=int)
        assert np.logical_and(faces >= 0, faces < len(pos)).all()
        return Data(pos=pos, faces=faces)

    def get_mean_std(self, sub_name=''):
        """ Computes Mean and Standard Deviation from Training Data.
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        n_input_channels: N
        (here N=5)
        """
        mean_std_cache = osp.join(self.processed_dir,
                                  sub_name + 'mean_std_cache.p')
        if not osp.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # no augmentations during m/std computation
            num_aug = self.num_aug
            self.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for j, mesh_data in enumerate(self):
                if j % 500 == 0:
                    print('{} of {}'.format(j, self.size))
                features = mesh_data['edge_attr']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (j + 1)
            std = std / (j + 1)
            transform_dict = {'mean': mean[:, np.newaxis],
                              'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.n_input_channels = transform_dict['ninput_channels']


class MeshCnnClassificationDataset(MeshCnnBaseDataset):
    r"""This class creates a classification dataset for MeshCNN networks:
    `"MeshCNN: A Network with an Edge" <https://arxiv.org/abs/1809.05910>`_
    paper.
    The dataset should be organized in folders according to class names. Each
    class folder should have 'train' and 'test' folders, where on these folders
    it should contain the .obj files. After the first pre-processing the class
    will create 'cache' folders which contain pre-processed .npz files after
    data augmentations.
    Args:
        root (str): Root folder for dataset.
        name (str): dataset name (supported are: shrec16, cubes)
        n_input_edges (int): Number of input edges of mesh. It should be the
                             bigger than or equal to highest number of input
                             edges in the dataset.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train dataset, if 'False' it will return the
                               test dataset. Default is 'True'.
        transform (MeshCNNPrepare, optional): meshcnn prepare transform. This
                                              transfrom will create the actual
                                              mesh_data and perform
                                              augmentations.
                                              Default is MeshCNNPrepare default
    """

    def __init__(self, root: str, name: str, n_input_edges: int,
                 train: bool = True, transform=MeshCNNPrepare()):

        self.name = name
        if name == 'shrec16':
            dataset_url = 'https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz?dl=1'
        elif name == 'cubes':
            dataset_url = 'https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz?dl=1'
        else:
            raise ValueError("dataset name %s is not supported" % name)

        self.mesh_prepare_transform = transform
        self.n_input_edges = n_input_edges
        self.train = train
        self.num_aug = transform.num_aug
        self.dir = None
        self.classes = []
        self.class_to_idx = []
        self.paths_labels = []
        self.mean = 0
        self.std = 1
        self.n_input_channels = None
        self.n_classes = None
        self.size = None
        if self.train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        MeshCnnBaseDataset.__init__(self, root=root, dataset_url=dataset_url)

    @staticmethod
    def find_classes(data_dir):
        classes = [d for d in os.listdir(data_dir) if
                   osp.isdir(osp.join(data_dir, d))]
        classes.sort()
        class_to_idx = {classes[k]: k for k in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def is_mesh_file(filename):
        return any(filename.endswith(extension) for extension in '.obj')

    def download(self):
        path = download_url(self.dataset_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')

    def make_dataset_by_class(self, data_dir, class_to_idx, phase):
        meshes = []
        data_dir = osp.expanduser(data_dir)
        for target in sorted(os.listdir(data_dir)):
            d = osp.join(data_dir, target)
            if not osp.isdir(d):
                continue
            for root, _, file_names in sorted(os.walk(d)):
                for file_name in sorted(file_names):
                    if self.is_mesh_file(file_name) and (
                            osp.basename(root) == phase):
                        path = osp.join(root, file_name)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes

    def process(self):
        self.dir = self.raw_dir + '\\' + \
                   [d for d in os.listdir(self.raw_dir) if
                    osp.isdir(osp.join(self.raw_dir, d))][0]
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths_labels = self.make_dataset_by_class(self.dir,
                                                       self.class_to_idx,
                                                       self.phase)
        self.n_classes = len(self.classes)
        self.size = len(self.paths_labels)
        self.get_mean_std()

        # Process all files - run augmentations and save for future use
        for path_label in self.paths_labels:
            path, label = path_label
            for num_aug in range(0, self.num_aug):
                filename, _ = osp.splitext(path)
                dir_name = osp.dirname(filename)
                prefix = osp.basename(filename)
                load_dir = osp.join(dir_name, 'cache')
                load_file = osp.join(load_dir,
                                     '%s_%03d.npz' % (prefix, num_aug))
                if not osp.isdir(load_dir):
                    os.makedirs(load_dir, exist_ok=True)
                if osp.exists(load_file):
                    continue
                else:
                    data = self.get_data_from_file(path)
                    self.mesh_prepare_transform(data, load_file)

    def len(self):
        return len(self.paths_labels)

    def get(self, idx):
        path = self.paths_labels[idx][0]
        label = self.paths_labels[idx][1]
        filename, _ = osp.splitext(path)
        dir_name = osp.dirname(filename)
        prefix = osp.basename(filename)
        load_dir = osp.join(dir_name, 'cache')
        if not osp.isdir(load_dir):
            os.makedirs(load_dir, exist_ok=True)
        load_file = osp.join(load_dir,
                             '%s_%03d.npz' % (
                                 prefix, np.random.randint(0, self.num_aug)))

        data = self.get_data_from_file(path)
        mesh_data = self.mesh_prepare_transform(data, load_file)
        mesh_data.label = label
        mesh_data.num_nodes = len(mesh_data.pos)
        mesh_data.edge_attr = pad(mesh_data.edge_attr, self.n_input_edges)
        mesh_data.edge_attr = (mesh_data.edge_attr - self.mean) / self.std
        return mesh_data


class MeshCnnSegmentationDataset(MeshCnnBaseDataset):
    r"""This class creates a segmentation dataset for MeshCNN networks:
    `"MeshCNN: A Network with an Edge" <https://arxiv.org/abs/1809.05910>`_
    paper.
    The dataset is splitted into 'train' and 'test' folders which contain the
    .obj files. In addition, 'seg' and 'sseg'
    folders should be in the dataset folder with .eseg and .seseg files - these
    files contain the data label and soft-
    label of the .obj files.
    After the first pre-processing the class will create 'cache' folder in the
    'train' folder which contain pre-
    processed .npz files after data augmentations.
    Args:
        root (str): Root folder for dataset.
        dataset_url (str): dataset URL link (see supported links in
                           MeshCnnBaseDataset)
        name (str): dataset name according to sub-dataset segmentation
                    name (supported are: coseg_aliens, coseg_chairs,
                    coseg_vases, human_seg).
        n_input_edges (int): Number of input edges of mesh. It should be the
                             bigger than or equal to highest number of input
                             edges in the dataset.
        train (bool, optional): Train or Test - if 'True' it will return
                               the train dataset, if 'False' it will return the
                               test dataset. Default is 'True'.
        transform (MeshCNNPrepare, optional): meshcnn prepare transform. This
                                              transfrom will create the actual
                                              mesh_data and perform
                                              augmentations.
                                              Default is MeshCNNPrepare default
    """

    def __init__(self, root: str, name: str, n_input_edges: int,
                 train: bool = True, transform=MeshCNNPrepare()):

        self.name = name
        if 'coseg_' in name:
            dataset_url = 'https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz?dl=1'
        elif name == 'human_seg':
            dataset_url = 'https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz?dl=1'
        else:
            raise ValueError("dataset name %s is not supported" % name)

        self.mesh_prepare_transform = transform
        self.n_input_edges = n_input_edges
        self.train = train
        self.num_aug = transform.num_aug
        self.dir = None
        self.paths = None
        self.offset = None
        self.classes = None
        self.sseg_paths = None
        self.seg_paths = None

        self.mean = 0
        self.std = 1
        self.n_input_channels = None
        self.n_classes = None
        self.size = None
        if self.train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        MeshCnnBaseDataset.__init__(self, root=root, dataset_url=dataset_url)

    @staticmethod
    def is_mesh_file(filename):
        return any(filename.endswith(extension) for extension in '.obj')

    def make_dataset(self, path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if self.is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes

    def download(self):
        path = download_url(self.dataset_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir,
                                   osp.splitext(os.path.basename(path))[
                                       0] + seg_ext)
            assert (os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def read_seg(seg):
        seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
        return seg_labels

    def read_sseg(self, sseg_file):
        sseg_labels = self.read_seg(sseg_file)
        sseg_labels = np.array(sseg_labels > 0, dtype=int)
        return sseg_labels

    def get_n_segs(self, classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, self.read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    def process(self):
        self.dir = osp.join(self.raw_dir, self.name)
        self.paths = self.make_dataset(osp.join(self.dir, self.phase))
        self.seg_paths = self.get_seg_files(self.paths,
                                            os.path.join(self.dir, 'seg'),
                                            seg_ext='.eseg')
        self.sseg_paths = self.get_seg_files(self.paths,
                                             os.path.join(self.dir, 'sseg'),
                                             seg_ext='.seseg')
        self.classes, self.offset = self.get_n_segs(
            os.path.join(self.dir, 'classes.txt'), self.seg_paths)
        self.n_classes = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std(self.name + '_')

        # Process all files - run augmentations and save for future use
        for path in self.paths:
            for num_aug in range(0, self.num_aug):
                filename, _ = osp.splitext(path)
                dir_name = osp.dirname(filename)
                prefix = osp.basename(filename)
                load_dir = osp.join(dir_name, 'cache')
                load_file = osp.join(load_dir,
                                     '%s_%03d.npz' % (prefix, num_aug))
                if not osp.isdir(load_dir):
                    os.makedirs(load_dir, exist_ok=True)
                if osp.exists(load_file):
                    continue
                else:
                    data = self.get_data_from_file(path)
                    self.mesh_prepare_transform(data, load_file)

    def len(self):
        return len(self.paths)

    def get(self, idx):
        path = self.paths[idx]
        label = self.read_seg(self.seg_paths[idx]) - self.offset
        label = pad(label, self.n_input_edges, val=-1, dim=0)
        soft_label = self.read_sseg(self.sseg_paths[idx])
        soft_label = pad(soft_label, self.n_input_edges, val=-1, dim=0)

        filename, _ = osp.splitext(path)
        dir_name = osp.dirname(filename)
        prefix = osp.basename(filename)
        load_dir = osp.join(dir_name, 'cache')
        load_file = osp.join(load_dir,
                             '%s_%03d.npz' % (
                                 prefix, np.random.randint(0, self.num_aug)))

        data = self.get_data_from_file(path)
        mesh_data = self.mesh_prepare_transform(data, load_file)
        mesh_data.label = label
        mesh_data.soft_label = soft_label
        mesh_data.num_nodes = len(mesh_data.pos)
        mesh_data.edge_attr = pad(mesh_data.edge_attr, self.n_input_edges)
        mesh_data.edge_attr = (mesh_data.edge_attr - self.mean) / self.std
        return mesh_data
