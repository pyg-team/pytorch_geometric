import os
import os.path as osp
import shutil
from itertools import chain
from xml.dom import minidom

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)

try:
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    models = None
    T = None
    Image = None


class PascalVOCKeypoints(InMemoryDataset):
    r"""The Pascal VOC 2011 dataset with Berkely annotations of keypoints from
    the `"Poselets: Body Part Detectors Trained Using 3D Human Pose
    Annotations" <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/
    human/ poselets_iccv09.pdf>`_ paper, containing 0 to 23 keypoints per
    example over 20 categories.
    The dataset is pre-filtered to exclude difficult, occluded and truncated
    objects.
    The keypoints contain interpolated features from a pre-trained VGG16 model
    on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

    Args:
        root (string): Root directory where the dataset should be saved.
        category (string): The category of the images (one of
            :obj:`"Aeroplane"`, :obj:`"Bicycle"`, :obj:`"Bird"`,
            :obj:`"Boat"`, :obj:`"Bottle"`, :obj:`"Bus"`, :obj:`"Car"`,
            :obj:`"Cat"`, :obj:`"Chair"`, :obj:`"Diningtable"`, :obj:`"Dog"`,
            :obj:`"Horse"`, :obj:`"Motorbike"`, :obj:`"Person"`,
            :obj:`"Pottedplant"`, :obj:`"Sheep"`, :obj:`"Sofa"`,
            :obj:`"Train"`, :obj:`"TVMonitor"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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
    image_url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2011/'
                 'VOCtrainval_25-May-2011.tar')
    annotation_url = ('https://www2.eecs.berkeley.edu/Research/Projects/CS/'
                      'vision/shape/poselets/voc2011_keypoints_Feb2012.tgz')
    # annotation_url = 'http://www.roemisch-drei.de/pascal_annotations.tar'
    # split_url = 'http://cvgl.stanford.edu/projects/ucn/voc2011_pairs.npz'
    split_url = ('https://github.com/Thinklab-SJTU/PCA-GM/raw/master/data/'
                 'PascalVOC/voc2011_pairs.npz')

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    def __init__(self, root, category, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.category = category.lower()
        assert self.category in self.categories
        super(PascalVOCKeypoints, self).__init__(root, transform,
                                                 pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.category.capitalize(), 'processed')

    @property
    def raw_file_names(self):
        return ['images', 'annotations', 'splits.npz']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.image_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)
        image_path = osp.join(self.raw_dir, 'TrainVal', 'VOCdevkit', 'VOC2011')
        os.rename(image_path, osp.join(self.raw_dir, 'images'))
        shutil.rmtree(osp.join(self.raw_dir, 'TrainVal'))

        path = download_url(self.annotation_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'splits.npz'))

    def process(self):
        if models is None or T is None or Image is None:
            raise ImportError('Package `torchvision` could not be found.')

        splits = np.load(osp.join(self.raw_dir, 'splits.npz'),
                         allow_pickle=True)
        category_idx = self.categories.index(self.category)
        train_split = list(splits['train'])[category_idx]
        test_split = list(splits['test'])[category_idx]

        image_path = osp.join(self.raw_dir, 'images', 'JPEGImages')
        info_path = osp.join(self.raw_dir, 'images', 'Annotations')
        annotation_path = osp.join(self.raw_dir, 'annotations')

        labels = {}

        vgg16_outputs = []

        def hook(module, x, y):
            vgg16_outputs.append(y)

        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        vgg16.features[20].register_forward_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_hook(hook)  # relu5_1

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_set, test_set = [], []
        for i, name in enumerate(chain(train_split, test_split)):
            filename = '_'.join(name.split('/')[1].split('_')[:-1])
            idx = int(name.split('_')[-1].split('.')[0]) - 1

            path = osp.join(info_path, '{}.xml'.format(filename))
            obj = minidom.parse(path).getElementsByTagName('object')[idx]

            trunc = obj.getElementsByTagName('truncated')[0].firstChild.data
            occ = obj.getElementsByTagName('occluded')
            occ = '0' if len(occ) == 0 else occ[0].firstChild.data
            diff = obj.getElementsByTagName('difficult')[0].firstChild.data

            if bool(int(trunc)) or bool(int(occ)) or bool(int(diff)):
                continue

            if self.category == 'person' and int(filename[:4]) > 2008:
                continue

            xmin = float(obj.getElementsByTagName('xmin')[0].firstChild.data)
            xmax = float(obj.getElementsByTagName('xmax')[0].firstChild.data)
            ymin = float(obj.getElementsByTagName('ymin')[0].firstChild.data)
            ymax = float(obj.getElementsByTagName('ymax')[0].firstChild.data)
            box = (xmin, ymin, xmax, ymax)

            dom = minidom.parse(osp.join(annotation_path, name))
            keypoints = dom.getElementsByTagName('keypoint')
            poss, ys = [], []
            for keypoint in keypoints:
                label = keypoint.attributes['name'].value
                if label not in labels:
                    labels[label] = len(labels)
                ys.append(labels[label])
                x = float(keypoint.attributes['x'].value)
                y = float(keypoint.attributes['y'].value)
                poss += [x, y]
            y = torch.tensor(ys, dtype=torch.long)
            pos = torch.tensor(poss, dtype=torch.float).view(-1, 2)

            if pos.numel() == 0:
                continue  # These examples do not make any sense anyway...

            # Add a small offset to the bounding because some keypoints lay
            # outside the bounding box intervals.
            box = (min(pos[:, 0].min().floor().item(), box[0]) - 16,
                   min(pos[:, 1].min().floor().item(), box[1]) - 16,
                   max(pos[:, 0].max().ceil().item(), box[2]) + 16,
                   max(pos[:, 1].max().ceil().item(), box[3]) + 16)

            # Rescale keypoints.
            pos[:, 0] = (pos[:, 0] - box[0]) * 256.0 / (box[2] - box[0])
            pos[:, 1] = (pos[:, 1] - box[1]) * 256.0 / (box[3] - box[1])

            path = osp.join(image_path, '{}.jpg'.format(filename))
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB').crop(box)
                img = img.resize((256, 256), resample=Image.BICUBIC)

            img = transform(img)

            data = Data(img=img, pos=pos, y=y, name=filename)

            if i < len(train_split):
                train_set.append(data)
            else:
                test_set.append(data)

        data_list = list(chain(train_set, test_set))
        imgs = [data.img for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()

            with torch.no_grad():
                vgg16(batch_img.to(self.device))

            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear',
                                 align_corners=False)

            for j in range(out1.size(0)):
                data = data_list[i * self.batch_size + j]
                idx = data.pos.round().long().clamp(0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                data.img = None
                data.x = torch.cat([x_1.t(), x_2.t()], dim=-1)
            del out1
            del out2

        if self.pre_filter is not None:
            train_set = [data for data in train_set if self.pre_filter(data)]
            test_set = [data for data in test_set if self.pre_filter(data)]

        if self.pre_transform is not None:
            train_set = [self.pre_transform(data) for data in train_set]
            test_set = [self.pre_transform(data) for data in test_set]

        torch.save(self.collate(train_set), self.processed_paths[0])
        torch.save(self.collate(test_set), self.processed_paths[1])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__, len(self),
                                            self.category)
