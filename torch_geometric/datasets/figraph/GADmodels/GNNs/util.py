import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, \
    balanced_accuracy_score, confusion_matrix, average_precision_score
import torch
from torch_geometric.data import Data
import pandas as pd


def KS(y_true, y_proba):
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic

def GM(y_true, y_pred):
    gmean = 1.0
    labels = sorted(list(set(y_true)))
    for label in labels:
        recall = (y_pred[y_true == label]).mean()
        gmean = gmean * recall
    return gmean ** (1 / len(labels))

def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

def evaluate(loss, labels, y_probs, epo, params, threshold=0.5):
    accuracy_list = []
    recall_list = []
    precision_list = []
    fpr_list = []
    f1_list = []
    roc_auc_list = []
    ks_list = []

    balanced_accuracy_list = []
    recall_macro_list = []
    precision_macro_list = []
    f1_macro_arithmetic_list = []
    f1_macro_harmonic_list = []
    mauc_list = []
    gm_list = []
    auprc_list = []

    gnn_pred_list = y_probs.argmax(dim=1).cpu().numpy().tolist()

    if labels.is_cuda:
        labels = labels.cpu().numpy()
        conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
        gmean_gnn = conf_gmean(conf_gnn)
    else:
        conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
        gmean_gnn = conf_gmean(conf_gnn)
    # y_preds = np.array([1 if i > threshold else 0 for i in y_probs])
    y_preds = y_probs.argmax(dim=1).cpu().detach().numpy()
    y_probs = y_probs[:, 1].cpu().detach().numpy()
    accuracy_list.append(accuracy_score(labels, y_preds))
    recall_list.append(recall_score(labels, y_preds, average='binary', pos_label=1))
    precision_list.append(precision_score(labels, y_preds, average='binary', pos_label=1))
    fpr_list.append((y_preds[labels == 0] == 1).mean())
    f1_list.append(f1_score(labels, y_preds, average='binary', pos_label=1))
    roc_auc_list.append(roc_auc_score(labels, y_probs))
    ks_list.append(KS(labels, y_probs))
    auprc_list.append(average_precision_score(labels, y_probs, pos_label=1))


    balanced_accuracy_list.append(balanced_accuracy_score(labels, y_preds))

    recall_macro = recall_score(labels, y_preds, average='macro')
    recall_macro_list.append(recall_macro)

    precision_macro = precision_score(labels, y_preds, average='macro')
    precision_macro_list.append(precision_macro)

    f1_macro_arithmetic_list.append(f1_score(labels, y_preds, average='macro'))
    # print(f1_score(labels, y_preds, average='macro'))
    f1_macro_harmonic = 2 * recall_macro * precision_macro / (recall_macro + precision_macro)
    f1_macro_harmonic_list.append(f1_macro_harmonic)

    mauc_list.append(roc_auc_score(labels, y_probs, average='macro', multi_class='ovo'))
    gm_list.append(GM(labels, y_preds))
    print(f"     GNN auc: {roc_auc_list[0]:.4f}",
          f"    GNN ks: {ks_list[0]:.4f}", f"    GNN gmean: {gmean_gnn:.4f}       {gm_list[0]:.4f}")
    print(f"macro GNN F1: {f1_score(labels, y_preds, average='macro') :.4f}", f"    GNN Recall: {recall_macro :.4f}",
          f"     GNN precision: {precision_macro:.4f}")
    epoch_list = [epo]
    loss_list = loss
    params_list = params

    indicator = np.vstack(
        [np.array(accuracy_list), np.array(recall_list),
         np.array(precision_list), np.array(fpr_list),
         np.array(f1_list), np.array(roc_auc_list), np.array(ks_list),np.array(auprc_list),

         np.array(balanced_accuracy_list), np.array(recall_macro_list),
         np.array(precision_macro_list), np.array(f1_macro_arithmetic_list),
         np.array(f1_macro_harmonic_list), np.array(mauc_list),
         np.array(gm_list),
         gmean_gnn,

         np.array(epoch_list),
         loss_list,
         params_list
         ])

    scores = pd.DataFrame(indicator.T,
                          columns=['Accuracy', 'Recall', 'Precision',
                                   'FPR', 'F1', 'ROC_AUC', 'KS','AUPRC',

                                   'Balanced_Accuracy', 'Recall_macro',
                                   'precision_macro', 'F1_macro_arithmetic',
                                   'F1_macro_harmonic', 'MAUC', 'GM','Gmean_gnn',

                                   'epoch', 'loss', 'Parmmeters'])

    return scores


from torch_geometric.nn import GATConv

class GAT_1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT_1, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


from torch_geometric.nn import SAGEConv

class GraphSAGE_1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE_1, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 16)
        self.conv2 = SAGEConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def load_data(year_list):
    # data = pd.read_csv("./data/已经离散化数据特征.csv")
    data = pd.read_csv("./data/已经离散化数据特征-Nips.csv")
    data = data[data['Year'].isin(year_list)]
    StkcdYear = data['StkcdYear']

    data.drop(columns=['Stkcd'], inplace=True)
    dis_cols = ['Audittyp', 'excessDebt_PropertyRightsNature', 'ProfitForecastTypeID','Audittyp_1','Audittyp_2','Audittyp_3','Audittyp_4','Audittyp_5','Audittyp_6','excessDebt_PropertyRightsNature_0','excessDebt_PropertyRightsNature_1','ProfitForecastTypeID_2','ProfitForecastTypeID_3','ProfitForecastTypeID_4','ProfitForecastTypeID_5','ProfitForecastTypeID_7','ProfitForecastTypeID_8','ProfitForecastTypeID_9','ProfitForecastTypeID_12','ProfitForecastTypeID_13']
    dis_cols = []
    num_cols = [col for col in list(data.columns) if col not in ['Year', 'label'] + dis_cols]
    # data = pd.get_dummies(data, columns=dis_cols)

    allcols = list(data.columns)
    allcols.remove('label')
    allcols.extend(['label'])
    data = data[allcols]

    trainset = pd.DataFrame.copy(
        data, deep=True)
    trainset.drop(columns=['Year'], axis=1, inplace=True)
    trainset[num_cols] = StandardScaler().fit_transform(trainset[num_cols])
    trainset['StkcdYear'] = StkcdYear
    return trainset

class GCN_1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_1, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)




def read_edges_from_files(file_paths):
    all_edges = []

    for file_path in file_paths:
        with open(file_path, 'r') as edge_file:
            edges = edge_file.readlines()
            # 处理每个文件中的边信息
            for edge in edges:
                start_node, end_node, edge_attr = edge.strip().split()
                all_edges.append((start_node, end_node, edge_attr))

    return all_edges


def map_nodes_to_indices(all_edges, feature_label_df, year_list):

    # 读取特征和标签文件
    feature_label_df = feature_label_df

    # 建立节点标识到索引的映射
    node_index_map = {}
    current_index = 0

    # 处理特征和标签文件
    node_features_labels = []
    node_feature_columns = feature_label_df.columns.difference(['StkcdYear', 'label'])

    for index, row in feature_label_df.iterrows():
        node = int(row['StkcdYear']) # 从 CSV 中读取的节点 id 可能是浮点数，需要转换为整数
        features = row[node_feature_columns]  # 前面的特征列
        label = row.iloc[-1]  # 最后一列是标签

        if node not in node_index_map:
            node_index_map[node] = current_index
            current_index += 1
        node_features_labels.append((node_index_map[node], features, label))

    # 处理连接关系文件
    mapped_edges = []
    for edge in all_edges:
        start_node, end_node, edge_attr = edge
        start_node = int(start_node)
        end_node = int(end_node)
        # if start_node not in node_index_map:
        #     node_index_map[start_node] = current_index
        #     current_index += 1
        # if end_node not in node_index_map:
        #     node_index_map[end_node] = current_index
        #     current_index += 1
        if start_node in node_index_map and end_node in node_index_map:
            mapped_edges.append((node_index_map[start_node], node_index_map[end_node], edge_attr))


    # train_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) in year_list[:7]]
    # val_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) == year_list[-2]]
    # test_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) == year_list[-1]]

    # return mapped_edges, node_features_labels , train_mask, val_mask, test_mask
    return mapped_edges, node_features_labels , [], [], []

def sigmoid_focal_loss(
        gnn_loss_xent: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: 0.82,
        gamma: 0,
        reduction: str = "mean",
) -> torch.Tensor:

    inputs = inputs.float()
    targets = targets.view([targets.size()[0], -1])
    targets = targets.float()
    p = torch.sigmoid(inputs)
    # p = inputs
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)

    loss = gnn_loss_xent * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    # print(loss)
    return loss
