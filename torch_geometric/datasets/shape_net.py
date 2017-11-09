import os

# import torch
from torch.utils.data import Dataset

# from ..graph.geometry import edges_from_faces
# from .utils.dir import make_dirs
# from .utils.ply import read_ply
# from .utils.download import download_url
# from .utils.extract import extract_tar


class ShapeNet(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'

    categories = {
        '02691156': 'airplaine',
        '02773838': 'bag',
        '02954340': 'cap',
        '02958343': 'car',
        '03001627': 'chair',
        '03261776': 'earphone',
        '03467517': 'guitar',
        '03624134': 'knife',
        '03636649': 'lamp',
        '03642806': 'laptop',
        '03790512': 'bike',
        '03797390': 'mug',
        '03948459': 'pistol',
        '04099429': 'rocket',
        '04225987': 'skateboard',
        '04379243': 'table',
    }

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(ShapeNet, self).__init__()

        print(self.url)
        self.root = os.path.expanduser(root)
        self.processed_folder = os.path.join(self.root, 'processed')

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()
