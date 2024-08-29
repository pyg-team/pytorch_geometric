import argparse
import csv
import datetime
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.graphsage import *
from src.layers import InterAgg, IntraAgg
from src.model import PCALayer
from src.utils import (
    load_all_data,
    load_data,
    normalize,
    pick_step,
    pos_neg_split,
    test_pcgnn,
)

"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler:
    def __init__(self, config):
        args = argparse.Namespace(**config)
        # load graph, feature, and label
        [homo, relation1, relation2,
         relation3], feat_data, labels = load_data(args.data_name,
                                                   prefix=args.data_dir)

        # train_test split
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.data_name == 'yelp':
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels, stratify=labels, train_size=args.train_ratio,
                random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
                random_state=2, shuffle=True)

        elif args.data_name == 'amazon':  # amazon
            # 0-3304 are unlabeled nodes
            index = list(range(3305, len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels[3305:], stratify=labels[3305:],
                train_size=args.train_ratio, random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
                random_state=2, shuffle=True)

        print(
            f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'
            +
            f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}'
        )
        print(f"Classification threshold: {args.thres}")
        print(f"Feature dimension: {feat_data.shape[1]}")

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)

        # if args.data == 'amazon':
        feat_data = normalize(feat_data)
        # train_feats = feat_data[np.array(idx_train)]
        # scaler = StandardScaler()
        # scaler.fit(train_feats)
        # feat_data = scaler.transform(feat_data)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

        # set input graph
        if args.model == 'SAGE' or args.model == 'GCN':
            adj_lists = homo
        else:
            adj_lists = [relation1, relation2, relation3]

        print(
            f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.'
        )

        self.args = args
        self.dataset = {
            'feat_data': feat_data,
            'labels': labels,
            'adj_lists': adj_lists,
            'homo': homo,
            'idx_train': idx_train,
            'idx_valid': idx_valid,
            'idx_test': idx_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
            'train_pos': train_pos,
            'train_neg': train_neg
        }

    def train(self):
        args = self.args
        features_all, y_all, dgl_all, adj_all, idx_train_all, idx_valid_all, idx_test_all, y_train_all, y_valid_all, y_test_all, array_all \
            = load_all_data()

        split_dicts = {
            0: {
                'train': [2014, 2016],
                'valid': 2017,
                'test': 2018
            },
            1: {
                'train': [2014, 2017],
                'valid': 2018,
                'test': 2019
            },
            2: {
                'train': [2014, 2018],
                'valid': 2019,
                'test': 2020
            },
            3: {
                'train': [2014, 2019],
                'valid': 2020,
                'test': 2021
            },
            4: {
                'train': [2014, 2020],
                'valid': 2021,
                'test': 2022
            }
        }
        # for times in range(5):
        for times in [0, 1, 2, 3]:
            loss_average = []
            result = []

            # feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
            # idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
            # idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset[
            #     'idx_test'], self.dataset['y_test']
            # initialize model input
            # features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
            # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
            # feat_data = normalize(feat_data)
            # features = torch.tensor(feat_data, dtype=torch.float32)
            # if args.cuda:
            #     features.cuda()

            # build one-layer models
            if args.model == 'PCGNN':
                intra1 = IntraAgg(features_all[0].shape[1], args.emb_size,
                                  args.rho, cuda=args.cuda)
                intra2 = IntraAgg(features_all[0].shape[1], args.emb_size,
                                  args.rho, cuda=args.cuda)
                intra3 = IntraAgg(features_all[0].shape[1], args.emb_size,
                                  args.rho, cuda=args.cuda)
                inter1 = InterAgg(features_all[0].shape[1], args.emb_size,
                                  inter=args.multi_relation, cuda=args.cuda)
            # elif args.model == 'SAGE':
            #     agg_sage = MeanAggregator(features, cuda=args.cuda)
            #     enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False,
            #                        cuda=args.cuda)
            # elif args.model == 'GCN':
            #     agg_gcn = GCNAggregator(features, cuda=args.cuda)
            #     enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True,
            #                          cuda=args.cuda)

            if args.model == 'PCGNN':
                gnn_model = PCALayer(2, inter1, args.alpha)
            # elif args.model == 'SAGE':
            #     # the vanilla GraphSAGE model as baseline
            #     enc_sage.num_samples = 5
            #     gnn_model = GraphSage(2, enc_sage)
            # elif args.model == 'GCN':
            #     gnn_model = GCN(2, enc_gcn)

            if args.cuda:
                gnn_model.cuda()

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, gnn_model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)

            timestamp = time.time()
            timestamp = datetime.datetime.fromtimestamp(
                int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
            dir_saver = args.save_dir + timestamp
            os.path.join(dir_saver, f'{args.data_name}_{args.model}.pkl')
            f1_mac_best, auc_best, ep_best = 0, 0, -1

            # train the model
            for epoch in range(200):
                loss_sum = 0.0

                for year in range(split_dicts[times]['train'][0],
                                  split_dicts[times]['train'][1] + 1):
                    labels = y_all[year - 2014]
                    idx_train = list(range(len(labels)))
                    y_train = y_all[year - 2014]
                    adj_lists = adj_all[year - 2014]
                    features = features_all[year - 2014]
                    train_pos, train_neg = pos_neg_split(idx_train, y_train)

                    sampled_idx_train = pick_step(idx_train, y_train,
                                                  adj_lists[0],
                                                  size=len(train_pos) * 2)

                    random.shuffle(sampled_idx_train)

                    num_batches = int(
                        len(sampled_idx_train) / args.batch_size) + 1

                    loss = 0.0
                    epoch_time = 0

                    # mini-batch training
                    for batch in range(num_batches):
                        start_time = time.time()
                        i_start = batch * args.batch_size
                        i_end = min((batch + 1) * args.batch_size,
                                    len(sampled_idx_train))
                        batch_nodes = sampled_idx_train[i_start:i_end]
                        batch_label = labels[np.array(batch_nodes)]
                        optimizer.zero_grad()
                        if args.cuda:
                            loss = gnn_model.loss(
                                batch_nodes,
                                Variable(torch.cuda.LongTensor(batch_label)))
                        else:
                            loss = gnn_model.loss(
                                batch_nodes,
                                Variable(torch.LongTensor(batch_label)),
                                adj_lists=adj_lists, features=features,
                                intra_list=[intra1, intra2,
                                            intra3], train_pos=train_pos)
                        loss.backward()
                        optimizer.step()
                        end_time = time.time()
                        epoch_time += end_time - start_time
                        loss += loss.item()

                    loss_sum += loss.item() / num_batches
                # loss_average.append(loss_sum/3)
                loss_average.append(loss_sum /
                                    (split_dicts[times]['train'][1] -
                                     split_dicts[times]['train'][0] + 1))
                # Valid the model for every $valid_epoch$ epoch
                # if args.model == 'SAGE' or args.model == 'GCN':
                #     print("Valid at epoch {}".format(epoch))
                #     f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model,
                #                                                                    args.batch_size, args.thres)
                #     if auc_val > auc_best:
                #         f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                #         if not os.path.exists(dir_saver):
                #             os.makedirs(dir_saver)
                #         print('  Saving model ...')
                #         torch.save(gnn_model.state_dict(), path_saver)
                # else:
                # print("Valid at epoch {}".format(epoch))

                y_train = y_all[split_dicts[times]['valid'] - 2014]
                idx_train = list(range(len(y_train)))
                train_pos, train_neg = pos_neg_split(idx_train, y_train)
                a = test_pcgnn(
                    range(len(y_all[split_dicts[times]['valid'] - 2014])),
                    y_all[split_dicts[times]['valid'] - 2014], gnn_model,
                    args.batch_size, thres=args.thres,
                    features=features_all[split_dicts[times]['valid'] - 2014],
                    adj_lists=adj_all[split_dicts[times]['valid'] - 2014],
                    intra_list=[intra1, intra2, intra3
                                ], train_pos=train_pos, epoch=epoch, params=2)
                if epoch == 0:
                    result = a
                else:
                    result = pd.concat([result, a], axis=0)

                y_train = y_all[split_dicts[times]['test'] - 2014]
                idx_train = list(range(len(y_train)))
                train_pos, train_neg = pos_neg_split(idx_train, y_train)
                a = test_pcgnn(
                    range(len(y_all[split_dicts[times]['test'] - 2014])),
                    y_all[split_dicts[times]['test'] - 2014], gnn_model,
                    args.batch_size, thres=args.thres,
                    features=features_all[split_dicts[times]['test'] - 2014],
                    adj_lists=adj_all[split_dicts[times]['test'] - 2014],
                    intra_list=[intra1, intra2, intra3
                                ], train_pos=train_pos, epoch=epoch, params=2)
                result = pd.concat([result, a], axis=0)
            file_name = 'result/result_Findata_Nips' + str(
                times) + '_' + 'PC-GNN' + '.csv'
            result.to_csv(file_name)
            # 将列表写入CSV文件
            file_name = 'result/result_Findata_Nips_train_loss_' + str(
                times) + '_' + 'PC-GNN' + '.csv'
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train_loss_average'])
                for i, value in enumerate(loss_average):
                    writer.writerow([i, value])

                    # if auc_val > auc_best:
                    #     f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                    #     if not os.path.exists(dir_saver):
                    #         os.makedirs(dir_saver)
                    #     print('  Saving model ...')
                    #     torch.save(gnn_model.state_dict(), path_saver)

                # print("Restore model from epoch {}".format(ep_best))
                # print("Model path: {}".format(path_saver))
                # gnn_model.load_state_dict(torch.load(path_saver))
                # if args.model == 'SAGE' or args.model == 'GCN':
                #     f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model,
                #                                                                         args.batch_size, args.thres)
                # else:
                #     f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model,
                #                                                                          args.batch_size, args.thres)
                # return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
        # args = self.args
        # feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
        # idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
        # idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset[
        #     'idx_test'], self.dataset['y_test']
        # # initialize model input
        # # features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        # # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        # features = torch.tensor(feat_data, dtype=torch.float32)
        # if args.cuda:
        #     features.cuda()
        #
        # # build one-layer models
        # if args.model == 'PCGNN':
        #     intra1 = IntraAgg(feat_data.shape[1], args.emb_size, args.rho,
        #                       cuda=args.cuda)
        #     intra2 = IntraAgg(feat_data.shape[1], args.emb_size, args.rho,
        #                       cuda=args.cuda)
        #     intra3 = IntraAgg(feat_data.shape[1], args.emb_size, args.rho,
        #                       cuda=args.cuda)
        #     inter1 = InterAgg(feat_data.shape[1], args.emb_size,
        #                       inter=args.multi_relation, cuda=args.cuda)
        # elif args.model == 'SAGE':
        #     agg_sage = MeanAggregator(features, cuda=args.cuda)
        #     enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False,
        #                        cuda=args.cuda)
        # elif args.model == 'GCN':
        #     agg_gcn = GCNAggregator(features, cuda=args.cuda)
        #     enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True,
        #                          cuda=args.cuda)
        #
        # if args.model == 'PCGNN':
        #     gnn_model = PCALayer(2, inter1, args.alpha)
        # elif args.model == 'SAGE':
        #     # the vanilla GraphSAGE model as baseline
        #     enc_sage.num_samples = 5
        #     gnn_model = GraphSage(2, enc_sage)
        # elif args.model == 'GCN':
        #     gnn_model = GCN(2, enc_gcn)
        #
        # if args.cuda:
        #     gnn_model.cuda()
        #
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
        #                              weight_decay=args.weight_decay)
        #
        # timestamp = time.time()
        # timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
        # dir_saver = args.save_dir + timestamp
        # path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
        # f1_mac_best, auc_best, ep_best = 0, 0, -1
        #
        # # train the model
        # for epoch in range(args.num_epochs):
        #     sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'],
        #                                   size=len(self.dataset['train_pos']) * 2)
        #
        #     random.shuffle(sampled_idx_train)
        #
        #     num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
        #
        #     loss = 0.0
        #     epoch_time = 0
        #
        #     # mini-batch training
        #     for batch in range(num_batches):
        #         start_time = time.time()
        #         i_start = batch * args.batch_size
        #         i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
        #         batch_nodes = sampled_idx_train[i_start:i_end]
        #         batch_label = self.dataset['labels'][np.array(batch_nodes)]
        #         optimizer.zero_grad()
        #         if args.cuda:
        #             loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)),
        #                                   features=features, adj_lists=adj_lists, intra_list=[intra1, intra2, intra3]
        #                                   , train_pos=self.dataset['train_pos'])
        #         else:
        #             loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)),
        #                                   features=features, adj_lists=adj_lists, intra_list=[intra1, intra2, intra3]
        #                                   , train_pos=self.dataset['train_pos'])
        #         loss.backward()
        #         optimizer.step()
        #         end_time = time.time()
        #         epoch_time += end_time - start_time
        #         loss += loss.item()
        #
        #     print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
        #
        #     # Valid the model for every $valid_epoch$ epoch
        #     if epoch % args.valid_epochs == 0:
        #         if args.model == 'SAGE' or args.model == 'GCN':
        #             print("Valid at epoch {}".format(epoch))
        #             f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model,
        #                                                                            args.batch_size, args.thres)
        #             if auc_val > auc_best:
        #                 f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
        #                 if not os.path.exists(dir_saver):
        #                     os.makedirs(dir_saver)
        #                 print('  Saving model ...')
        #                 torch.save(gnn_model.state_dict(), path_saver)
        #         else:
        #             print("Valid at epoch {}".format(epoch))
        #             f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = \
        #                 test_pcgnn(idx_valid, y_valid, gnn_model,
        #                            args.batch_size, thres= args.thres,adj_lists=adj_lists,
        #                            epoch=epoch, features=features, intra_list=[intra1, intra2, intra3]
        #                            , train_pos=self.dataset['train_pos'], params=1)
        #             if auc_val > auc_best:
        #                 f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
        #                 if not os.path.exists(dir_saver):
        #                     os.makedirs(dir_saver)
        #                 print('  Saving model ...')
        #                 torch.save(gnn_model.state_dict(), path_saver)
        #
        # print("Restore model from epoch {}".format(ep_best))
        # print("Model path: {}".format(path_saver))
        # gnn_model.load_state_dict(torch.load(path_saver))
        # if args.model == 'SAGE' or args.model == 'GCN':
        #     f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model,
        #                                                                         args.batch_size, args.thres)
        # else:
        #     f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model,
        #                                                                          args.batch_size, args.thres)
        # return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
