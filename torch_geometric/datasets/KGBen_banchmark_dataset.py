from torch_geometric.data import InMemoryDataset
import pandas as pd
import zipfile
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
''' list of KGBen benchmark datasets '''
KGBen_datasets_dic={
    "MAG42M_PV_FG":"http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_FG.zip",
    "MAG42M_PV_d1h1":"http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_d1h1.zip",
    "DBLP15M_PV_FG":"http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_FG.zip",
    "DBLP15M_PV_d1h1":"http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_d1h1.zip",
    "YAGO_FM200":"http://206.12.102.56/CodsData/KGNET/KGBen/YAGO/YAGO_FM200.zip",
    "YAGO_Star200":"http://206.12.102.56/CodsData/KGNET/YAGO/DBLP/YAGO_Star200.zip",
}
''' KGBen benchmark dataset in-memory loader'''
class KGBen_banchmark_dataset(InMemoryDataset):
    def __init__(self, name, root = '/media/hussein/UbuntuData/GithubRepos/pytorch_geometric/data/',numofClasses=349, transform=None, pre_transform=None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 
        self.datasets_path=root
        self.name = name ## original name, e.g., ogbn-proteins

        if meta_dict is None:
            # self.dir_name = self.datasets_path+'_'.join(name.split('-'))
            self.dir_name = osp.join(self.datasets_path , name)
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = self.datasets_path #root
            self.root = osp.join(root, self.dir_name)
            
            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            # if not self.name in master:
            #     error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
            #     error_mssg += 'Available datasets are as follows:\n'
            #     error_mssg += '\n'.join(master.keys())
            #     raise ValueError(error_mssg)
            # self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.

        # if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
        #     print(self.name + ' has been updated.')
        #     if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
        #         shutil.rmtree(self.root)
        # if osp.exists(self.root):
        #     shutil.rmtree(self.root)

        # self.meta_info = {'url': self.dir_name + ".zip"}
        self.meta_info = {'url': KGBen_datasets_dic[self.name]}
        self.meta_info['download_name'] = {'url': self.dir_name + ".zip"}
        # f = zipfile.ZipFile(self.dir_name + ".zip", 'r')
        self.download_name = self.dir_name   ## name of downloaded file, e.g., tox21
        self.meta_info['add_inverse_edge']=True
        self.meta_info['has_node_attr'] = True
        self.meta_info['has_edge_attr'] = False
        self.meta_info['split'] = 'time'
        self.meta_info['additional node files']='node_year'
        self.meta_info['additional edge files'] = 'edge_reltype'
        self.meta_info['is hetero'] = 'True'
        self.meta_info['binary'] = 'False'
        self.meta_info['eval metric']='acc'
        self.meta_info['num classes']=numofClasses
        self.meta_info['num tasks']='1'
        self.meta_info['task type']='multiclass classification'
        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'
        super(KGBen_banchmark_dataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        url =  self.meta_info['url']
        if str(url).startswith("http")==False:
            path =url
            extract_zip(path, self.original_root)
            # os.unlink(path) # delete  file
            if self.download_name.split("/")[-1]!=self.root.split("/")[-1]:
                shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        elif decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            # shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

        else:
            data = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip', header = None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

if __name__ == '__main__':
    pyg_dataset = KGBen_banchmark_dataset(name = 'MAG42M_PV_FG')
    print(pyg_dataset[0])
    split_index = pyg_dataset.get_idx_split()
    # print(split_index)
    