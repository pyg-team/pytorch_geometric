from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class RECT_L(torch.nn.Module):
    r"""The GNN model RECT (or more specifically its supervised part RECT-L) from the TKDE20 paper
    `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`.
    The model has three simple steps: 1) calculate class semantic knowledge; 2) train a gnn model; 3) detach to get embedding results
        For an example of using the RECT model, see
        `examples/rect.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        rect.py>`_.
    Args:
        in_feats (int): Size of each input sample.
        n_hidden (int): Size of each output sample.
        dropout (float): The dropout probability.
    """
    def __init__(self, in_feats, n_hidden, dropout):
        super(RECT_L, self).__init__()
        self.gcn = GCNConv(in_feats, n_hidden, cached=True, normalize=False)
        self.fc = torch.nn.Linear(n_hidden, in_feats)
        self.dropout = dropout
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        
    def forward(self, inputs, edge_index, edge_attr):
        self.edge_index, self.edge_weight = edge_index, edge_attr
        h_1 = self.gcn(inputs,self.edge_index, self.edge_weight)
        h_1 = F.dropout(h_1, p=self.dropout, training=self.training)
        preds = self.fc(h_1)
        return preds
    
    # Detach the return variables
    def embed(self, inputs):
        h_1 = self.gcn(inputs, self.edge_index, self.edge_weight)
        return h_1.detach()
   
    def __repr__(self):
        return '{}(layer_1={})'.format(self.__class__.__name__, self.gcn) + '\n' + '{}(layer_2={})'.format(self.__class__.__name__, self.fc)
    
def remove_unseen_classes_from_training(train_mask, labels, removed_classes):
    ''' Remove the unseen classes from training data to get the zero-shot (i.e., completely imbalanced) label setting
        Input: train_mask, labels, removed_classes
        Output: train_mask_zs: the bool list indicating seen classes
    '''
    train_mask_zs = train_mask.clone()
    for i in range(train_mask_zs.numel()):
        if train_mask_zs[i]==1 and (labels[i].item() in removed_classes):
            train_mask_zs[i]=0
    return train_mask_zs

def get_label_attributes(train_mask_zs, nodeids, labellist, features):
    ''' Get the class-center (semanic knowledge) of each seen class.
        Suppose a node i is labeled as c, then attribute[c] += node_i_attribute, finally mean(attribute[c])
        Input: train_mask_zs, nodeids, labellist, features
        Output: label_attribute{}: label -> average_labeled_node_features (class centers)
    '''
    label_attribute_nodes = defaultdict(list)
    for nodeid, label in zip(nodeids, labellist):
        label_attribute_nodes[int(label)].append(int(nodeid))
    label_attribute = {}
    for label in label_attribute_nodes.keys():
        nodes = label_attribute_nodes[int(label)]
        label_attribute[int(label)] = np.mean(features[nodes, :], axis=0)
    return label_attribute

def get_labeled_nodes_label_attribute(train_mask_zs, labels, features):
    ''' Replace the original labels by their class-centers.
        For each label c in the training set, we first get label's attribute, then set res[i, :] = label_attribute[c]
        Input: train_mask_zs, labels, features
        Output: y_{semantic} [l, ft]: tensor
    '''
    X = torch.LongTensor(range(features.shape[0]))
    nodeids = []
    labellist = []
    for i in X[train_mask_zs].numpy().tolist():
        nodeids.append(str(i))
    for i in labels[train_mask_zs].cpu().numpy().tolist():
        labellist.append(str(i))

    # 1. get the semantic knowledge (class centers) of all seen classes
    label_attribute = get_label_attributes(train_mask_zs=train_mask_zs, nodeids=nodeids, labellist=labellist, features=features.cpu().numpy())
    
    # 2. replace original labels by their class centers (semantic knowledge)
    y_semantic = np.zeros([len(nodeids), features.shape[1]])
    for i, label in enumerate(labellist):
        y_semantic[i, :] = label_attribute[int(label)]
        
    return torch.FloatTensor(y_semantic)

def svd_feature(features, d=200):
    ''' Dimension reduction'''
    if(features.shape[1] <= d): return features
    U, S, Vh = torch.linalg.svd(features)
    return torch.mm(U[:, 0:d], torch.diag(S[0:d]))
