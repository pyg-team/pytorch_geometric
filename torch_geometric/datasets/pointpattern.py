import os
import scipy.io as sio
import numpy as np
import gdown
import zipfile
import torch
from torch_geometric.data import Data

def PointPattern(phi=0.3):
    """
    PointPattern is for 3-classification dataset of point distribtuion graphs, stemming from statistical mechanics.

    Parameters
    ----------
    phi : hardness of classification task
        DESCRIPTION. The default is 0.3. Other options: 0.35, 0.4.

    Returns
    -------
    pointpattern : PYTORCH_GEOMETRIC.DATASETS
        Datasets inclusing 15000 graph sampling.
        Attributes: pointpattern.num_edge      --  average number of edges
                    pointpattern.num_feature   --  number of features for each graph
                    pointpattern.num_node      --  average number of nodes
                    pointpattern.num_classes   --  number of classes
                    pointpattern.num_graph     --  number of graph samples

    """
    num_graph = 15000
    if phi==0.3:
        ld_dir = 'hpr_phi03' + '_' + str(num_graph) + '/'
        url = 'https://drive.google.com/uc?id=1C3ciJsteqsKFVGF8JI8-KnXhe4zY41Ss'
        output = 'hpr_phi03' + '_' + str(num_graph) + '.zip'
    if phi==0.4:
        ld_dir = 'hpr_phi04' + '_' + str(num_graph) + '/'
        url = 'https://drive.google.com/uc?id=1rsTh09FzGxHculBVrYyl5tOHD9mxqc0G'
        output = 'hpr_phi04' + '_' + str(num_graph) + '.zip'
    if phi==0.35:
        ld_dir = 'hpr_phi035' + '_' + str(num_graph) + '/'
        url = 'https://drive.google.com/uc?id=16pI974P8WzanBUPrMHIaGfeSLoksviBk'
        output = 'hpr_phi035' + '_' + str(num_graph) + '.zip'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()
    #os.remove(output)
    # load edge_index
    ld_edge_index = ld_dir + 'graph' + str(num_graph) + '_edge_index' + '.mat'
    edge_index = sio.loadmat(ld_edge_index)
    edge_index = edge_index['edge_index'][0]
    # load feature
    ld_feature = ld_dir + 'graph' + str(num_graph) + '_feature' + '.mat'
    feature = sio.loadmat(ld_feature)
    feature = feature['feature'][0]
    # load label
    ld_label = ld_dir + 'graph' + str(num_graph) + '_label' + '.mat'
    label = sio.loadmat(ld_label)
    label = label['label']
    ## store edge, feature and label into a graph, in format of "torch_geometric.datasets.Data"
    pointpattern = list()
    num_edge = 0
    num_feature = 0
    num_node = 0
    num_classes = 3
    num_graph = edge_index.shape[0]
    
    for i in range(num_graph):
        # extract edge index, turn to tensor
        edge_index_1 = np.array(edge_index[i][:,0:2],dtype=np.int)
        edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long)
        # number of edges
        num_edge = num_edge + edge_index_1.shape[0]
        # extract feature, turn to tensor
        feature_1 = torch.tensor(np.array(feature[i],dtype=np.int), dtype=torch.float)
        # number of nodes
        num_node = num_node + feature_1.shape[0]
        # number of features
        if i==0:
            num_feature = feature_1.shape[1]
        # extract label, turn to tensor
        label_1 = torch.tensor(label[i],dtype=torch.long)
        # put edge, feature, label together to form graph information in "Data" format
        data_1 = Data(x=feature_1, edge_index=edge_index_1.t().contiguous(), y=label_1)
        pointpattern.append(data_1)
    
    ave_num_edge = num_edge*1.0/num_graph
    ave_num_node = num_node*1.0/num_graph
    
    return pointpattern, ave_num_edge, ave_num_node, num_feature, num_classes, num_graph