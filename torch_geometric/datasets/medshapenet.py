from typing import Callable, List, Optional
import urllib3
import glob
import os
import os.path as osp
import numpy as np

import torch

from torch_geometric.data import Data, InMemoryDataset

from MedShapeNet import MedShapeNet as msn
msn_instance = msn()

class MedShapeNet(InMemoryDataset):
  r"""The MedShapeNet datasets from the `"MedShapeNet -- A Large-Scale Dataset
    of 3D Medical Shapes for Computer Vision"
    <https://arxiv.org/abs/2308.16139>`_ paper,
    containing 8 different type of structures (classes).

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (str): Root directory where the dataset should be saved.
        size (int): Number of invividual 3D structures to download per type (classes).
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
    """
  def __init__(
        self,
        root: str,
        size: int = 100,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.size = size
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)


  @property
  def raw_file_names(self) -> List[str]:
      return [
          '3DTeethSeg', 'CoronaryArteries', 'FLARE', 'KITS', 'PULMONARY',
          'SurgicalInstruments', 'ThoracicAorta_Saitta', 'ToothFairy'
      ]

  @property
  def processed_file_names(self) -> List[str]:
      return ['training.pt','test.pt']

  @property
  def raw_paths(self):
    r"""The absolute filepaths that must be present in order to skip
    downloading."""
    files = self.raw_file_names
    return [osp.join(self.raw_dir, f) for f in files] or osp.join(self.raw_dir, files)

  def download(self):
    pool = urllib3.HTTPConnectionPool("medshapenet.ddns.net", maxsize=50)

    list_of_datasets = msn_instance.datasets(False)
    list_of_datasets = list(filter(lambda x: x not in ['medshapenetcore/ASOCA','medshapenetcore/AVT','medshapenetcore/AutoImplantCraniotomy','medshapenetcore/FaceVR'], list_of_datasets))

    for dataset in list_of_datasets:
      self.newpath = self.root + '/' + dataset.split("/")[1]
      if not os.path.exists(self.newpath):
        os.makedirs(self.newpath)
      stl_files = msn_instance.dataset_files(dataset, '.stl')
      stl_files = stl_files[:self.size] if self.train else stl_files[-self.size:]
      for stl_file in stl_files:
        msn_instance.download_stl_as_numpy(bucket_name = dataset, stl_file = stl_file, output_dir = self.newpath, print_output=False)

  def process(self) -> None:
    self.save(self.process_set('train'), self.processed_paths[0])
    self.save(self.process_set('test'), self.processed_paths[1])

  def process_set(self, set_name: str) -> List[Data]:
    categories = glob.glob(osp.join(self.root, '', '*'))
    categories = list(filter(lambda x: x not in [(self.root + 'raw')], categories))
    data_list = []
    for target, category in enumerate(categories):
      files = glob.glob(osp.join(self.root, category, '*.npz'))
      for file in files:
        data = np.load(file)
        pre_data_list = Data(
            pos = torch.tensor(data["vertices"], dtype=torch.float),
            face = torch.tensor(data["faces"], dtype=torch.long).t().contiguous()
        )
        pre_data_list.y = torch.tensor([target], dtype=torch.long)
        data_list.append(pre_data_list)

    if self.pre_filter is not None:
        data_list = [d for d in data_list if self.pre_filter(d)]

    if self.pre_transform is not None:
        data_list = [self.pre_transform(d) for d in data_list]

    return data_list


