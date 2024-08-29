import math
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):
    def __init__(self, feature_dim, embed_dim, inter='GNN', cuda=True):
        """Initialize the inter-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples in training set
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregators used by each single-relation graph
        :param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param cuda: whether to use GPU
        """
        super().__init__()

        self.dropout = 0.6

        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda

        # initial filtering thresholds
        self.thresholds = [0.5, 0.5, 0.5]

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(
            torch.FloatTensor(self.embed_dim * 3 + self.feat_dim,
                              self.embed_dim))
        init.xavier_uniform_(self.weight)

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []
        self.thresholds_log = [self.thresholds]
        self.relation_score_log = []

    def forward(self, nodes, labels, features, train_pos, adj_lists, intraggs,
                train_flag=True):
        """:param nodes: a list of batch node ids
        :param labels: a list of batch node labels
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """
        self.features = features

        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        # self.intra_agg1.cuda = cuda
        # self.intra_agg2.cuda = cuda
        # self.intra_agg3.cuda = cuda
        self.train_pos = train_pos
        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(set.union(*to_neighs[0]),
                                 set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(
                torch.cuda.LongTensor(list(unique_nodes)))
            pos_features = self.features(
                torch.cuda.LongTensor(list(self.train_pos)))
        else:
            batch_features = self.features[list(unique_nodes)]
            pos_features = self.features[list(self.train_pos)]
        batch_scores = self.label_clf(batch_features)
        pos_scores = self.label_clf(pos_features)
        id_mapping = {
            node_id: index
            for node_id, index in zip(unique_nodes, range(len(unique_nodes)))
        }

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list for each batch node and relation
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        # assign label-aware scores to neighbor nodes for each batch node and relation
        r1_scores = [
            batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2)
            for to_neigh in r1_list
        ]
        r2_scores = [
            batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2)
            for to_neigh in r2_list
        ]
        r3_scores = [
            batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2)
            for to_neigh in r3_list
        ]

        # count the number of neighbors kept for aggregation for each batch node and relation
        r1_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list
        ]
        r2_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list
        ]
        r3_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list
        ]

        # intra-aggregation steps for each relation
        # Eq. (8) in the paper
        r1_feats, r1_scores = self.intra_agg1.forward(
            nodes, labels, r1_list, center_scores, r1_scores, pos_scores,
            r1_sample_num_list, train_flag, features, train_pos)
        r2_feats, r2_scores = self.intra_agg2.forward(
            nodes, labels, r2_list, center_scores, r2_scores, pos_scores,
            r2_sample_num_list, train_flag, features, train_pos)
        r3_feats, r3_scores = self.intra_agg3.forward(
            nodes, labels, r3_list, center_scores, r3_scores, pos_scores,
            r3_sample_num_list, train_flag, features, train_pos)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features[list(index)]
        # self_feats = self.features[index, :] #第四次

        # number of nodes in a batch
        len(nodes)

        # concat the intra-aggregated embeddings from each relation
        # Eq. (9) in the paper
        cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats),
                              dim=1)

        combined = F.relu(cat_feats.mm(self.weight).t())

        return combined, center_scores


class IntraAgg(nn.Module):
    def __init__(self, feat_dim, embed_dim, rho, cuda=False):
        """Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples in training set
        :param rho: the ratio of the oversample neighbors for the minority class
        :param cuda: whether to use GPU
        """
        super().__init__()

        self.cuda = cuda
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.rho = rho
        self.weight = nn.Parameter(
            torch.FloatTensor(2 * self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, batch_labels, to_neighs_list, batch_scores,
                neigh_scores, pos_scores, sample_list, train_flag, features,
                train_pos):
        """Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
        :param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
        :param train_flag: indicates whether in training or testing mode
        :param sample_list: the number of neighbors kept for each batch node in one relation
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """
        self.features = features
        self.train_pos = train_pos
        # filer neighbors under given relation in the train mode
        if train_flag:
            samp_neighs, samp_scores = choose_step_neighs(
                batch_scores, batch_labels, neigh_scores, to_neighs_list,
                pos_scores, self.train_pos, sample_list, self.rho)
        else:
            samp_neighs, samp_scores = choose_step_test(
                batch_scores, neigh_scores, to_neighs_list, sample_list)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [
            unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh
        ]
        row_indices = [
            i for i in range(len(samp_neighs))
            for _ in range(len(samp_neighs[i]))
        ]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)  # mean aggregator
        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
            embed_matrix = self.features(
                torch.LongTensor(unique_nodes_list).cuda())
        else:
            self_feats = self.features[nodes]
            embed_matrix = self.features[unique_nodes_list]
        agg_feats = mask.mm(embed_matrix)  # single relation aggregator
        cat_feats = torch.cat((self_feats, agg_feats),
                              dim=1)  # concat with last layer
        to_feats = F.relu(cat_feats.mm(self.weight))
        return to_feats, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list,
                       minor_scores, minor_list, sample_list, sample_rate):
    """Choose step for neighborhood sampling
    :param center_scores: the label-aware scores of batch nodes
    :param center_labels: the label of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param minor_scores: the label-aware scores for nodes of minority class in one relation
    :param minor_list: minority node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :para sample_rate: the ratio of the oversample neighbors for the minority class
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff_neigh = torch.abs(center_score_neigh -
                                     neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(
            score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [
                neighs_indices[n] for n in selected_neigh_indices[:num_sample]
            ]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(
                center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(
                score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([
                minor_list[n] for n in selected_minor_indices[:num_oversample]
            ])
            selected_score_diff.extend(
                sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
    """Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_scores: the average neighbor distances for each relation after filtering
    """
    samp_neighs = []
    samp_scores = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0,
                                                   descending=False)
        selected_indices = sorted_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [
                neighs_indices[n] for n in selected_indices[:num_sample]
            ]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores
