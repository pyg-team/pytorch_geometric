from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
# import zipfile
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import Data
import urllib.request as ur
import errno
import zipfile
# from ogb.utils.url import decide_download, download_url,extract_zip
# from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
# from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
''' list of KGBen benchmark datasets '''
KGBen_datasets_dic={
    "MAG42M_PV_FG":"http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_FG.zip",
    "MAG42M_PV_d1h1":"http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_d1h1.zip",
    "DBLP15M_PV_FG":"http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_FG.zip",
    "DBLP15M_PV_d1h1":"http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_d1h1.zip",
    "YAGO_FM200":"http://206.12.102.56/CodsData/KGNET/KGBen/YAGO/YAGO_FM200.zip",
    "YAGO_Star200":"http://206.12.102.56/CodsData/KGNET/YAGO/DBLP/YAGO_Star200.zip",
}
GBFACTOR = float(1 << 30)
''' helper function copied from the ogb library to read and load heterogeneous OGB format dataset in a zipped file format'''
class ogb_helper_functions:
    @staticmethod
    def decide_download(url):
        d = ur.urlopen(url)
        size = int(d.info()["Content-Length"])/GBFACTOR

        ### confirm if larger than 1GB
        if size > 1:
            return input("This will download %.2fGB. Will you proceed? (y/N)\n" % (size)).lower() == "y"
        else:
            return True
    @staticmethod
    def makedirs(path):
        try:
            os.makedirs(osp.expanduser(osp.normpath(path)))
        except OSError as e:
            if e.errno != errno.EEXIST and osp.isdir(path):
                raise e
    @staticmethod
    def download_url(url, folder, log=True):
        r"""Downloads the content of an URL to a specific folder.
        Args:
            url (string): The url.
            folder (string): The folder.
            log (bool, optional): If :obj:`False`, will not print anything to the
                console. (default: :obj:`True`)
        """

        filename = url.rpartition('/')[2]
        path = osp.join(folder, filename)

        if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
            if log:
                print('Using exist file', filename)
            return path

        if log:
            print('Downloading', url)

        ogb_helper_functions.makedirs(folder)
        data = ur.urlopen(url)

        size = int(data.info()["Content-Length"])

        chunk_size = 1024*1024
        num_iter = int(size/chunk_size) + 2

        downloaded_size = 0

        try:
            with open(path, 'wb') as f:
                pbar = tqdm(range(num_iter))
                for i in pbar:
                    chunk = data.read(chunk_size)
                    downloaded_size += len(chunk)
                    pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                    f.write(chunk)
        except:
            if os.path.exists(path):
                 os.remove(path)
            raise RuntimeError('Stopped downloading due to interruption.')


        return path
    @staticmethod
    def maybe_log(path, log=True):
        if log:
            print('Extracting', path)
    @staticmethod
    def extract_zip(path, folder, log=True):
        r"""Extracts a zip archive to a specific folder.
        Args:
            path (string): The path to the tar archive.
            folder (string): The folder.
            log (bool, optional): If :obj:`False`, will not print anything to the
                console. (default: :obj:`True`)
        """
        ogb_helper_functions.maybe_log(path, log)
        with zipfile.ZipFile(path, 'r') as f:
            f.extractall(folder)
    @staticmethod
    def read_npz_dict(path):
        tmp = np.load(path)
        dict = {}
        for key in tmp.keys():
            dict[key] = tmp[key]
        del tmp
        return dict
    @staticmethod
    def read_node_label_hetero(raw_dir):
        import pandas as pd
        df = pd.read_csv(osp.join(raw_dir, 'nodetype-has-label.csv.gz'))
        label_dict = {}
        for nodetype in df.keys():
            has_label = df[nodetype].values[0]
            if has_label:
                label_dict[nodetype] = pd.read_csv(osp.join(raw_dir, 'node-label', nodetype, 'node-label.csv.gz'), compression='gzip', header = None).values

        if len(label_dict) == 0:
            raise RuntimeError('No node label file found.')

        return label_dict
    @staticmethod
    def read_nodesplitidx_split_hetero(split_dir):
        import pandas as pd
        df = pd.read_csv(osp.join(split_dir, 'nodetype-has-split.csv.gz'))
        train_dict = {}
        valid_dict = {}
        test_dict = {}
        for nodetype in df.keys():
            has_label = df[nodetype].values[0]
            if has_label:
                train_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
                valid_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
                test_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        if len(train_dict) == 0:
            raise RuntimeError('No split file found.')

        return train_dict, valid_dict, test_dict
    @staticmethod
    def read_binary_heterograph_raw(raw_dir, add_inverse_edge=False):
        '''
        raw_dir: path to the raw directory
        add_inverse_edge (bool): whether to add inverse edge or not

        return: graph_list, which is a list of heterogeneous graphs.
        Each graph is a dictionary, containing the following keys:
        - edge_index_dict
            edge_index_dict[(head, rel, tail)] = edge_index for (head, rel, tail)

        - edge_feat_dict
            edge_feat_dict[(head, rel, tail)] = edge_feat for (head, rel, tail)

        - node_feat_dict
            node_feat_dict[nodetype] = node_feat for nodetype

        - num_nodes_dict
            num_nodes_dict[nodetype] = num_nodes for nodetype

        * edge_feat_dict and node_feat_dict are optional: if a graph does not contain it, we will simply have None.

        We can also have additional node/edge features. For example,
        - edge_**
        - node_**

        '''

        if add_inverse_edge:
            raise RuntimeError('add_inverse_edge is depreciated in read_binary')

        print('Loading necessary files...')
        print('This might take a while.')

        # loading necessary files
        try:
            num_nodes_dict = ogb_helper_functions.read_npz_dict(osp.join(raw_dir, 'num_nodes_dict.npz'))
            tmp = ogb_helper_functions.read_npz_dict(osp.join(raw_dir, 'num_edges_dict.npz'))
            num_edges_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
            del tmp
            tmp = ogb_helper_functions.read_npz_dict(osp.join(raw_dir, 'edge_index_dict.npz'))
            edge_index_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
            del tmp

            ent_type_list = sorted(list(num_nodes_dict.keys()))
            triplet_type_list = sorted(list(num_edges_dict.keys()))

            num_graphs = len(num_nodes_dict[ent_type_list[0]])

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        # storing node and edge features
        # mapping from the name of the features to feat_dict
        node_feat_dict_dict = {}
        edge_feat_dict_dict = {}

        for filename in os.listdir(raw_dir):
            if '.npz' not in filename:
                continue
            if filename in ['num_nodes_dict.npz', 'num_edges_dict.npz', 'edge_index_dict.npz']:
                continue

            # do not read target label information here
            if '-label.npz' in filename:
                continue

            feat_name = filename.split('.')[0]

            if 'node_' in feat_name:
                feat_dict = ogb_helper_functions.read_npz_dict(osp.join(raw_dir, filename))
                node_feat_dict_dict[feat_name] = feat_dict
            elif 'edge_' in feat_name:
                tmp = ogb_helper_functions.read_npz_dict(osp.join(raw_dir, filename))
                feat_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
                del tmp
                edge_feat_dict_dict[feat_name] = feat_dict
            else:
                raise RuntimeError(
                    f"Keys in graph object should start from either \'node_\' or \'edge_\', but found \'{feat_name}\'.")

        graph_list = []
        num_nodes_accum_dict = {ent_type: 0 for ent_type in ent_type_list}
        num_edges_accum_dict = {triplet: 0 for triplet in triplet_type_list}

        print('Processing graphs...')
        for i in tqdm(range(num_graphs)):

            graph = dict()

            ### set up default atribute
            graph['edge_index_dict'] = {}
            graph['num_nodes_dict'] = {}

            for feat_name in node_feat_dict_dict.keys():
                graph[feat_name] = {}

            for feat_name in edge_feat_dict_dict.keys():
                graph[feat_name] = {}

            if not 'edge_feat_dict' in graph:
                graph['edge_feat_dict'] = None

            if not 'node_feat_dict' in graph:
                graph['node_feat_dict'] = None

            ### handling edge
            for triplet in triplet_type_list:
                edge_index = edge_index_dict[triplet]
                num_edges = num_edges_dict[triplet][i]
                num_edges_accum = num_edges_accum_dict[triplet]

                ### add edge_index
                graph['edge_index_dict'][triplet] = edge_index[:, num_edges_accum:num_edges_accum + num_edges]

                ### add edge feature
                for feat_name in edge_feat_dict_dict.keys():
                    if triplet in edge_feat_dict_dict[feat_name]:
                        feat = edge_feat_dict_dict[feat_name][triplet]
                        graph[feat_name][triplet] = feat[num_edges_accum: num_edges_accum + num_edges]

                num_edges_accum_dict[triplet] += num_edges

            ### handling node
            for ent_type in ent_type_list:
                num_nodes = num_nodes_dict[ent_type][i]
                num_nodes_accum = num_nodes_accum_dict[ent_type]

                ### add node feature
                for feat_name in node_feat_dict_dict.keys():
                    if ent_type in node_feat_dict_dict[feat_name]:
                        feat = node_feat_dict_dict[feat_name][ent_type]
                        graph[feat_name][ent_type] = feat[num_nodes_accum: num_nodes_accum + num_nodes]

                graph['num_nodes_dict'][ent_type] = num_nodes
                num_nodes_accum_dict[ent_type] += num_nodes

            graph_list.append(graph)

        return graph_list
    ### reading raw files from a directory.
    ### for heterogeneous graph
    @staticmethod
    def read_csv_heterograph_raw(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[]):
        '''
        raw_dir: path to the raw directory
        add_inverse_edge (bool): whether to add inverse edge or not

        return: graph_list, which is a list of heterogeneous graphs.
        Each graph is a dictionary, containing the following keys:
        - edge_index_dict
            edge_index_dict[(head, rel, tail)] = edge_index for (head, rel, tail)

        - edge_feat_dict
            edge_feat_dict[(head, rel, tail)] = edge_feat for (head, rel, tail)

        - node_feat_dict
            node_feat_dict[nodetype] = node_feat for nodetype

        - num_nodes_dict
            num_nodes_dict[nodetype] = num_nodes for nodetype

        * edge_feat_dict and node_feat_dict are optional: if a graph does not contain it, we will simply have None.

        We can also have additional node/edge features. For example,
        - edge_reltype_dict
            edge_reltype_dict[(head, rel, tail)] = edge_reltype for (head, rel, tail)

        - node_year_dict
            node_year_dict[nodetype] = node_year

        '''
        import pandas as pd
        print('Loading necessary files...')
        print('This might take a while.')

        # loading necessary files
        try:
            num_node_df = pd.read_csv(osp.join(raw_dir, 'num-node-dict.csv.gz'), compression='gzip')
            num_node_dict = {nodetype: num_node_df[nodetype].astype(np.int64).tolist() for nodetype in num_node_df.keys()}
            nodetype_list = sorted(list(num_node_dict.keys()))

            ## read edge_dict, num_edge_dict
            triplet_df = pd.read_csv(osp.join(raw_dir, 'triplet-type-list.csv.gz'), compression='gzip', header=None)
            triplet_list = sorted([(head, relation, tail) for head, relation, tail in
                                   zip(triplet_df[0].tolist(), triplet_df[1].tolist(), triplet_df[2].tolist())])

            edge_dict = {}
            num_edge_dict = {}

            for triplet in triplet_list:
                subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))

                edge_dict[triplet] = pd.read_csv(osp.join(subdir, 'edge.csv.gz'), compression='gzip',
                                                 header=None).values.T.astype(np.int64)
                num_edge_dict[triplet] = \
                pd.read_csv(osp.join(subdir, 'num-edge-list.csv.gz'), compression='gzip', header=None).astype(np.int64)[
                    0].tolist()

            # check the number of graphs coincide
            assert (len(num_node_dict[nodetype_list[0]]) == len(num_edge_dict[triplet_list[0]]))

            num_graphs = len(num_node_dict[nodetype_list[0]])

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        node_feat_dict = {}
        for nodetype in nodetype_list:
            subdir = osp.join(raw_dir, 'node-feat', nodetype)

            try:
                node_feat = pd.read_csv(osp.join(subdir, 'node-feat.csv.gz'), compression='gzip', header=None).values
                if 'int' in str(node_feat.dtype):
                    node_feat = node_feat.astype(np.int64)
                else:
                    # float
                    node_feat = node_feat.astype(np.float32)

                node_feat_dict[nodetype] = node_feat
            except FileNotFoundError:
                pass

        edge_feat_dict = {}
        for triplet in triplet_list:
            subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))

            try:
                edge_feat = pd.read_csv(osp.join(subdir, 'edge-feat.csv.gz'), compression='gzip', header=None).values
                if 'int' in str(edge_feat.dtype):
                    edge_feat = edge_feat.astype(np.int64)
                else:
                    # float
                    edge_feat = edge_feat.astype(np.float32)

                edge_feat_dict[triplet] = edge_feat

            except FileNotFoundError:
                pass

        additional_node_info = {}
        # e.g., additional_node_info['node_year'] = node_feature_dict for node_year
        for additional_file in additional_node_files:
            additional_feat_dict = {}
            assert (additional_file[:5] == 'node_')

            for nodetype in nodetype_list:
                subdir = osp.join(raw_dir, 'node-feat', nodetype)

                try:
                    node_feat = pd.read_csv(osp.join(subdir, additional_file + '.csv.gz'), compression='gzip',
                                            header=None).values
                    if 'int' in str(node_feat.dtype):
                        node_feat = node_feat.astype(np.int64)
                    else:
                        # float
                        node_feat = node_feat.astype(np.float32)

                    assert (len(node_feat) == sum(num_node_dict[nodetype]))

                    additional_feat_dict[nodetype] = node_feat

                except FileNotFoundError:
                    pass

            additional_node_info[additional_file] = additional_feat_dict

        additional_edge_info = {}
        # e.g., additional_edge_info['edge_reltype'] = edge_feat_dict for edge_reltype
        for additional_file in additional_edge_files:
            assert (additional_file[:5] == 'edge_')
            additional_feat_dict = {}
            for triplet in triplet_list:
                subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))

                try:
                    edge_feat = pd.read_csv(osp.join(subdir, additional_file + '.csv.gz'), compression='gzip',
                                            header=None).values
                    if 'int' in str(edge_feat.dtype):
                        edge_feat = edge_feat.astype(np.int64)
                    else:
                        # float
                        edge_feat = edge_feat.astype(np.float32)

                    assert (len(edge_feat) == sum(num_edge_dict[triplet]))

                    additional_feat_dict[triplet] = edge_feat

                except FileNotFoundError:
                    pass

            additional_edge_info[additional_file] = additional_feat_dict

        graph_list = []
        num_node_accum_dict = {nodetype: 0 for nodetype in nodetype_list}
        num_edge_accum_dict = {triplet: 0 for triplet in triplet_list}

        print('Processing graphs...')
        for i in tqdm(range(num_graphs)):

            graph = dict()

            ### set up default atribute
            graph['edge_index_dict'] = {}
            graph['edge_feat_dict'] = {}
            graph['node_feat_dict'] = {}
            graph['num_nodes_dict'] = {}

            ### set up additional node/edge attributes
            for key in additional_node_info.keys():
                graph[key] = {}

            for key in additional_edge_info.keys():
                graph[key] = {}

            ### handling edge
            for triplet in triplet_list:
                edge = edge_dict[triplet]
                num_edge = num_edge_dict[triplet][i]
                num_edge_accum = num_edge_accum_dict[triplet]

                if add_inverse_edge:
                    ### add edge_index
                    # duplicate edge
                    duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum + num_edge], 2, axis=1)
                    duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]
                    duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]
                    graph['edge_index_dict'][triplet] = duplicated_edge

                    ### add default edge feature
                    if len(edge_feat_dict) > 0:
                        # if edge_feat exists for some triplet
                        if triplet in edge_feat_dict:
                            graph['edge_feat_dict'][triplet] = np.repeat(
                                edge_feat_dict[triplet][num_edge:num_edge + num_edge], 2, axis=0)

                    else:
                        # if edge_feat is not given for any triplet
                        graph['edge_feat_dict'] = None

                    ### add additional edge feature
                    for key, value in additional_edge_info.items():
                        if triplet in value:
                            graph[key][triplet] = np.repeat(value[triplet][num_edge_accum: num_edge_accum + num_edge], 2,
                                                            axis=0)

                else:
                    ### add edge_index
                    graph['edge_index_dict'][triplet] = edge[:, num_edge_accum:num_edge_accum + num_edge]

                    ### add default edge feature
                    if len(edge_feat_dict) > 0:
                        # if edge_feat exists for some triplet
                        if triplet in edge_feat_dict:
                            graph['edge_feat_dict'][triplet] = edge_feat_dict[triplet][num_edge:num_edge + num_edge]

                    else:
                        # if edge_feat is not given for any triplet
                        graph['edge_feat_dict'] = None

                    ### add additional edge feature
                    for key, value in additional_edge_info.items():
                        if triplet in value:
                            graph[key][triplet] = value[triplet][num_edge_accum: num_edge_accum + num_edge]

                num_edge_accum_dict[triplet] += num_edge

            ### handling node
            for nodetype in nodetype_list:
                num_node = num_node_dict[nodetype][i]
                num_node_accum = num_node_accum_dict[nodetype]

                ### add default node feature
                if len(node_feat_dict) > 0:
                    # if node_feat exists for some node type
                    if nodetype in node_feat_dict:
                        graph['node_feat_dict'][nodetype] = node_feat_dict[nodetype][
                                                            num_node_accum:num_node_accum + num_node]

                else:
                    graph['node_feat_dict'] = None

                    ### add additional node feature
                for key, value in additional_node_info.items():
                    if nodetype in value:
                        graph[key][nodetype] = value[nodetype][num_node_accum: num_node_accum + num_node]

                graph['num_nodes_dict'][nodetype] = num_node
                num_node_accum_dict[nodetype] += num_node

            graph_list.append(graph)

        return graph_list
    ### reading raw files from a directory.
    ### npz ver
    ### for homogeneous graph
    @staticmethod
    def read_binary_graph_raw(raw_dir, add_inverse_edge = False):
        '''
        raw_dir: path to the raw directory
        add_inverse_edge (bool): whether to add inverse edge or not

        return: graph_list, which is a list of graphs.
        Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
        edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

        raw_dir must contain data.npz
        - edge_index
        - num_nodes_list
        - num_edges_list
        - node_** (optional, node_feat is the default node features)
        - edge_** (optional, edge_feat is the default edge features)
        '''

        if add_inverse_edge:
            raise RuntimeError('add_inverse_edge is depreciated in read_binary')

        print('Loading necessary files...')
        print('This might take a while.')
        data_dict = np.load(osp.join(raw_dir, 'data.npz'))

        edge_index = data_dict['edge_index']
        num_nodes_list = data_dict['num_nodes_list']
        num_edges_list = data_dict['num_edges_list']

        # storing node and edge features
        node_dict = {}
        edge_dict = {}

        for key in list(data_dict.keys()):
            if key == 'edge_index' or key == 'num_nodes_list' or key == 'num_edges_list':
                continue

            if key[:5] == 'node_':
                node_dict[key] = data_dict[key]
            elif key[:5] == 'edge_':
                edge_dict[key] = data_dict[key]
            else:
                raise RuntimeError(f"Keys in graph object should start from either \'node_\' or \'edge_\', but found \'{key}\'.")

        graph_list = []
        num_nodes_accum = 0
        num_edges_accum = 0

        print('Processing graphs...')
        for num_nodes, num_edges in tqdm(zip(num_nodes_list, num_edges_list), total=len(num_nodes_list)):

            graph = dict()

            graph['edge_index'] = edge_index[:, num_edges_accum:num_edges_accum+num_edges]

            for key, feat in edge_dict.items():
                graph[key] = feat[num_edges_accum:num_edges_accum+num_edges]

            if 'edge_feat' not in graph:
                graph['edge_feat'] =  None

            for key, feat in node_dict.items():
                graph[key] = feat[num_nodes_accum:num_nodes_accum+num_nodes]

            if 'node_feat' not in graph:
                graph['node_feat'] = None

            graph['num_nodes'] = num_nodes

            num_edges_accum += num_edges
            num_nodes_accum += num_nodes

            graph_list.append(graph)

        return graph_list
    ### reading raw files from a directory.
    ### for homogeneous graph
    @staticmethod
    def read_csv_graph_raw(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[]):
        '''
        raw_dir: path to the raw directory
        add_inverse_edge (bool): whether to add inverse edge or not

        return: graph_list, which is a list of graphs.
        Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
        edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

        additional_node_files and additional_edge_files must be in the raw directory.
        - The name should be {additional_node_file, additional_edge_file}.csv.gz
        - The length should be num_nodes or num_edges

        additional_node_files must start from 'node_'
        additional_edge_files must start from 'edge_'


        '''
        import pandas as pd
        print('Loading necessary files...')
        print('This might take a while.')
        # loading necessary files
        try:
            edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), compression='gzip', header=None).values.T.astype(
                np.int64)  # (2, num_edge) numpy array
            num_node_list = \
            pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), compression='gzip', header=None).astype(np.int64)[
                0].tolist()  # (num_graph, ) python list
            num_edge_list = \
            pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), compression='gzip', header=None).astype(np.int64)[
                0].tolist()  # (num_edge, ) python list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        try:
            node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), compression='gzip', header=None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                # float
                node_feat = node_feat.astype(np.float32)
        except FileNotFoundError:
            node_feat = None

        try:
            edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), compression='gzip', header=None).values
            if 'int' in str(edge_feat.dtype):
                edge_feat = edge_feat.astype(np.int64)
            else:
                # float
                edge_feat = edge_feat.astype(np.float32)

        except FileNotFoundError:
            edge_feat = None

        additional_node_info = {}
        for additional_file in additional_node_files:
            assert (additional_file[:5] == 'node_')

            # hack for ogbn-proteins
            if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
                os.rename(osp.join(raw_dir, 'species.csv.gz'), osp.join(raw_dir, 'node_species.csv.gz'))

            temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header=None).values

            if 'int' in str(temp.dtype):
                additional_node_info[additional_file] = temp.astype(np.int64)
            else:
                # float
                additional_node_info[additional_file] = temp.astype(np.float32)

        additional_edge_info = {}
        for additional_file in additional_edge_files:
            assert (additional_file[:5] == 'edge_')
            temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header=None).values

            if 'int' in str(temp.dtype):
                additional_edge_info[additional_file] = temp.astype(np.int64)
            else:
                # float
                additional_edge_info[additional_file] = temp.astype(np.float32)

        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0

        print('Processing graphs...')
        for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):

            graph = dict()

            ### handling edge
            if add_inverse_edge:
                ### duplicate edge
                duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum + num_edge], 2, axis=1)
                duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]
                duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]

                graph['edge_index'] = duplicated_edge

                if edge_feat is not None:
                    graph['edge_feat'] = np.repeat(edge_feat[num_edge_accum:num_edge_accum + num_edge], 2, axis=0)
                else:
                    graph['edge_feat'] = None

                for key, value in additional_edge_info.items():
                    graph[key] = np.repeat(value[num_edge_accum:num_edge_accum + num_edge], 2, axis=0)

            else:
                graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum + num_edge]

                if edge_feat is not None:
                    graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum + num_edge]
                else:
                    graph['edge_feat'] = None

                for key, value in additional_edge_info.items():
                    graph[key] = value[num_edge_accum:num_edge_accum + num_edge]

            num_edge_accum += num_edge

            ### handling node
            if node_feat is not None:
                graph['node_feat'] = node_feat[num_node_accum:num_node_accum + num_node]
            else:
                graph['node_feat'] = None

            for key, value in additional_node_info.items():
                graph[key] = value[num_node_accum:num_node_accum + num_node]

            graph['num_nodes'] = num_node
            num_node_accum += num_node

            graph_list.append(graph)
        return graph_list
    @staticmethod
    def read_graph_pyg(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[], binary=False):
        if binary:
            # npz
            graph_list = ogb_helper_functions.read_binary_graph_raw(raw_dir, add_inverse_edge)
        else:
            # csv
            graph_list = ogb_helper_functions.read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files,
                                            additional_edge_files=additional_edge_files)
        pyg_graph_list = []
        print('Converting graphs into PyG objects...')
        for graph in tqdm(graph_list):
            g = Data()
            g.num_nodes = graph['num_nodes']
            g.edge_index = torch.from_numpy(graph['edge_index'])

            del graph['num_nodes']
            del graph['edge_index']

            if graph['edge_feat'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_feat'])
                del graph['edge_feat']

            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
                del graph['node_feat']

            for key in additional_node_files:
                g[key] = torch.from_numpy(graph[key])
                del graph[key]

            for key in additional_edge_files:
                g[key] = torch.from_numpy(graph[key])
                del graph[key]

            pyg_graph_list.append(g)

        return pyg_graph_list

    @staticmethod
    def read_heterograph_pyg(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[],
                             binary=False):
        if binary:
            # npz
            graph_list = ogb_helper_functions.read_binary_heterograph_raw(raw_dir, add_inverse_edge)
        else:
            # csv
            graph_list = ogb_helper_functions.read_csv_heterograph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files,
                                                  additional_edge_files=additional_edge_files)

        pyg_graph_list = []

        print('Converting graphs into PyG objects...')

        for graph in tqdm(graph_list):
            g = Data()

            g.__num_nodes__ = graph['num_nodes_dict']
            g.num_nodes_dict = graph['num_nodes_dict']

            # add edge connectivity
            g.edge_index_dict = {}
            for triplet, edge_index in graph['edge_index_dict'].items():
                g.edge_index_dict[triplet] = torch.from_numpy(edge_index)

            del graph['edge_index_dict']

            if graph['edge_feat_dict'] is not None:
                g.edge_attr_dict = {}
                for triplet in graph['edge_feat_dict'].keys():
                    g.edge_attr_dict[triplet] = torch.from_numpy(graph['edge_feat_dict'][triplet])

                del graph['edge_feat_dict']

            if graph['node_feat_dict'] is not None:
                g.x_dict = {}
                for nodetype in graph['node_feat_dict'].keys():
                    g.x_dict[nodetype] = torch.from_numpy(graph['node_feat_dict'][nodetype])

                del graph['node_feat_dict']

            for key in additional_node_files:
                g[key] = {}
                for nodetype in graph[key].keys():
                    g[key][nodetype] = torch.from_numpy(graph[key][nodetype])

                del graph[key]

            for key in additional_edge_files:
                g[key] = {}
                for triplet in graph[key].keys():
                    g[key][triplet] = torch.from_numpy(graph[key][triplet])

                del graph[key]

            pyg_graph_list.append(g)

        return pyg_graph_list
################################################################################
''' KGBen benchmark dataset's in-memory loader'''
class KGBen_banchmark_dataset(InMemoryDataset):
    def __init__(self, name, root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/")) ,numofClasses=349, transform=None, pre_transform=None, meta_dict = None):
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
        import pandas as pd
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = ogb_helper_functions.read_nodesplitidx_split_hetero(path)
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
            ogb_helper_functions.extract_zip(path, self.original_root)
            # os.unlink(path) # delete  file
            if self.download_name.split("/")[-1]!=self.root.split("/")[-1]:
                shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        elif ogb_helper_functions.decide_download(url):
            path = ogb_helper_functions.download_url(url, self.original_root)
            ogb_helper_functions.extract_zip(path, self.original_root)
            os.unlink(path)
            # shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        import pandas as pd
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
            data = ogb_helper_functions.read_heterograph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]
            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = ogb_helper_functions.read_node_label_hetero(self.raw_dir)

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
            data = ogb_helper_functions.read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

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
    