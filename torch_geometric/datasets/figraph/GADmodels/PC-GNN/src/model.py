import torch
import torch.nn as nn
from torch.nn import init

"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class PCALayer(nn.Module):
    """One Pick-Choose-Aggregate layer
    """
    def __init__(self, num_classes, inter1, lambda_1):
        """Initialize the PC-GNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter1: the inter-relation aggregator that output the final embedding
        """
        super().__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()

        # the parameter to transform the final embedding
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, inter1.embed_dim))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1
        self.epsilon = 0.1

    def forward(self, nodes, labels, adj_lists, intra_list, features,
                train_pos, train_flag=True):
        embeds1, label_scores = self.inter1(nodes, labels, adj_lists=adj_lists,
                                            intraggs=intra_list,
                                            features=features,
                                            train_pos=train_pos,
                                            train_flag=train_flag)
        scores = self.weight.mm(embeds1)
        return scores.t(), label_scores

    def to_prob(self, nodes, labels, adj_lists, intra_list, features,
                train_pos, train_flag=True):
        gnn_logits, label_logits = self.forward(nodes, labels, adj_lists,
                                                intra_list, features,
                                                train_pos, train_flag)
        gnn_scores = torch.sigmoid(gnn_logits)
        label_scores = torch.sigmoid(label_logits)
        return gnn_scores, label_scores

    def loss(self, nodes, labels, adj_lists, intra_list, features, train_pos,
             train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, adj_lists,
                                                intra_list, features,
                                                train_pos, train_flag)
        # Simi loss, Eq. (7) in the paper
        label_loss = self.xent(label_scores, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        # the loss function of PC-GNN, Eq. (11) in the paper
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss
