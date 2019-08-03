import os
import os.path as osp
import shutil
import glob

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)


class WILLOWObjectClass(InMemoryDataset):
    r"""The WILLOW-ObjectClass dataset from the `"Learning Graphs to Match"
    <https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>`_ paper,
    containing 10 keypoints with pre-trained VGG16 features (:obj:`relu4_2`
    and :obj:`relu5_1`) of at least 40 images in each category.

    Args:
        root (string): Root directory where the dataset should be saved.
        category (string): The category of the images (one of :obj:`"Car"`,
            :obj:`"Duck"`, :obj:`"Face"`, :obj:`"Motorbike"`,
            :obj:`"Winebottle"`)
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
    url = ('http://www.di.ens.fr/willow/research/graphlearning/'
           'WILLOW-ObjectClass_dataset.zip')

    def __init__(self, root, category='Car', transform=None,
                 pre_transform=None, pre_filter=None):
        assert category in ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
        self.category = category
        super(WILLOWObjectClass, self).__init__(root, transform, pre_transform,
                                                pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.category.lower())

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.unlink(osp.join(self.root, 'README'))
        os.unlink(osp.join(self.root, 'demo_showAnno.m'))
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, 'WILLOW-ObjectClass'), self.raw_dir)

    def process(self):
        names = glob.glob(osp.join(self.raw_dir, self.category, '*.png'))
        names = sorted([name[:-4] for name in names])

        vgg16_outputs = []

        def hook(module, x, y):
            vgg16_outputs.append(y)

        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        vgg16.features[20].register_forward_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_hook(hook)  # relu5_1

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_list = []
        for name in names:
            pos = loadmat('{}.mat'.format(name))['pts_coord']
            x, y = torch.from_numpy(pos).to(torch.float)
            pos = torch.stack([x, y], dim=1)

            with open('{}.png'.format(name), 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = transform(img)
            size = img.size()[-2:]
            vgg16_outputs.clear()
            vgg16(img.unsqueeze(0))

            xs = []
            for out in vgg16_outputs:
                out = F.interpolate(out, size, mode='bilinear',
                                    align_corners=False)
                out = out.squeeze(0).permute(1, 2, 0)
                out = out[y.round().long(), x.round().long()]
                xs.append(out)

            data = Data(x=torch.cat(xs, dim=-1), pos=pos)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__, len(self),
                                            self.category)
