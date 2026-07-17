import os
import os.path as osp
from itertools import chain
from typing import Callable, Dict, List, Optional
from xml.dom import minidom

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.io import fs


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
        root (str): Root directory where the dataset should be saved.
        category (str): The category of the images (one of
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        device (str or torch.device, optional): The device to use for
            processing the raw data. If set to :obj:`None`, will utilize
            GPU-processing if available. (default: :obj:`None`)
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

    batch_size = 32

    def __init__(
        self,
        root: str,
        category: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.category = category.lower()
        assert self.category in self.categories
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.category.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['images', 'annotations', 'splits.npz']

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self) -> None:
        path = download_url(self.image_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)
        image_path = osp.join(self.raw_dir, 'TrainVal', 'VOCdevkit', 'VOC2011')
        os.rename(image_path, osp.join(self.raw_dir, 'images'))
        fs.rm(osp.join(self.raw_dir, 'TrainVal'))

        path = download_url(self.annotation_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'splits.npz'))

    def process(self) -> None:
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image

        splits = np.load(osp.join(self.raw_dir, 'splits.npz'),
                         allow_pickle=True)
        category_idx = self.categories.index(self.category)
        train_split = list(splits['train'])[category_idx]
        test_split = list(splits['test'])[category_idx]

        image_path = osp.join(self.raw_dir, 'images', 'JPEGImages')
        info_path = osp.join(self.raw_dir, 'images', 'Annotations')
        annotation_path = osp.join(self.raw_dir, 'annotations')

        labels: Dict[str, int] = {}

        vgg16_outputs = []

        def hook(module: torch.nn.Module, x: Tensor, y: Tensor) -> None:
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
            file_idx = int(name.split('_')[-1].split('.')[0]) - 1

            path = osp.join(info_path, f'{filename}.xml')
            obj = minidom.parse(path).getElementsByTagName('object')[file_idx]

            child = obj.getElementsByTagName('truncated')[0].firstChild
            assert child is not None
            trunc = child.data  # type: ignore

            elements = obj.getElementsByTagName('occluded')
            if len(elements) == 0:
                occ = '0'
            else:
                child = elements[0].firstChild
                assert child is not None
                occ = child.data  # type: ignore

            child = obj.getElementsByTagName('difficult')[0].firstChild
            diff = child.data  # type: ignore

            if bool(int(trunc)) or bool(int(occ)) or bool(int(diff)):
                continue

            if self.category == 'person' and int(filename[:4]) > 2008:
                continue

            child = obj.getElementsByTagName('xmin')[0].firstChild
            assert child is not None
            xmin = int(child.data)  # type: ignore

            child = obj.getElementsByTagName('xmax')[0].firstChild
            assert child is not None
            xmax = int(child.data)  # type: ignore

            child = obj.getElementsByTagName('ymin')[0].firstChild
            assert child is not None
            ymin = int(child.data)  # type: ignore

            child = obj.getElementsByTagName('ymax')[0].firstChild
            assert child is not None
            ymax = int(child.data)  # type: ignore

            box = (xmin, ymin, xmax, ymax)

            dom = minidom.parse(osp.join(annotation_path, name))
            keypoints = dom.getElementsByTagName('keypoint')
            poss, ys = [], []
            for keypoint in keypoints:
                label = keypoint.attributes['name'].value
                if label not in labels:
                    labels[label] = len(labels)
                ys.append(labels[label])
                _x = float(keypoint.attributes['x'].value)
                _y = float(keypoint.attributes['y'].value)
                poss += [_x, _y]
            y = torch.tensor(ys, dtype=torch.long)
            pos = torch.tensor(poss, dtype=torch.float).view(-1, 2)

            if pos.numel() == 0:
                continue  # These examples do not make any sense anyway...

            # Add a small offset to the bounding because some keypoints lay
            # outside the bounding box intervals.
            box = (
                min(int(pos[:, 0].min().floor()), box[0]) - 16,
                min(int(pos[:, 1].min().floor()), box[1]) - 16,
                max(int(pos[:, 0].max().ceil()), box[2]) + 16,
                max(int(pos[:, 1].max().ceil()), box[3]) + 16,
            )

            # Rescale keypoints.
            pos[:, 0] = (pos[:, 0] - box[0]) * 256.0 / (box[2] - box[0])
            pos[:, 1] = (pos[:, 1] - box[1]) * 256.0 / (box[3] - box[1])

            path = osp.join(image_path, f'{filename}.jpg')
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB').crop(box)
                img = img.resize((256, 256), resample=Image.Resampling.BICUBIC)

            img = transform(img)

            data = Data(img=img, pos=pos, y=y, name=filename)

            if i < len(train_split):
                train_set.append(data)
            else:
                test_set.append(data)

        data_list = list(chain(train_set, test_set))
        imgs = [data.img for data in data_list]
        loader: DataLoader = DataLoader(
            dataset=imgs,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
        )
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
                assert data.pos is not None
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

        self.save(train_set, self.processed_paths[0])
        self.save(test_set, self.processed_paths[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')
