import argparse
import csv
import os
import random
import time

from graphsage import *
from layers import *
from model import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='yelp',
                    help='The dataset name. [yelp, amazon]')
parser.add_argument('--model', type=str, default='CARE',
                    help='The model name. [CARE, SAGE]')
parser.add_argument(
    '--inter', type=str, default='GNN',
    help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2,
                    help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3,
                    help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64,
                    help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=31,
                    help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=3,
                    help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1,
                    help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2,
                    help='RL action step size')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'run on {args.data}')

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

# load graph, feature, and label
# [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)

# train_test split

# if args.data == 'yelp':
#     index = list(range(len(labels)))
#     idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60,
#                                                             random_state=2, shuffle=True)
# elif args.data == 'amazon':  # amazon
#     # 0-3304 are unlabeled nodes
#     index = list(range(3305, len(labels)))
#     idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
#                                                             test_size=0.60, random_state=2, shuffle=True)
print(
    f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.'
)

# for times in range(5):
for times in [4]:
    np.random.seed(args.seed)
    random.seed(args.seed)
    loss_average = []
    result = []

    # build one-layer models
    if args.model == 'CARE':
        intra1 = IntraAgg(features_all[0].shape[1], cuda=args.cuda)
        intra2 = IntraAgg(features_all[0].shape[1], cuda=args.cuda)
        intra3 = IntraAgg(features_all[0].shape[1], cuda=args.cuda)
        inter1 = InterAgg(features_all[0].shape[1], args.emb_size,
                          inter=args.inter, step_size=args.step_size,
                          cuda=args.cuda)
        gnn_model = OneLayerCARE(2, inter1, args.lambda_1)

    if args.cuda:
        gnn_model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
        weight_decay=args.lambda_2)

    for epoch in range(200):
        loss_sum = 0
        for year in range(split_dicts[times]['train'][0],
                          split_dicts[times]['train'][1] + 1):
            labels = y_all[year - 2014]
            idx_train = list(range(len(labels)))
            y_train = y_all[year - 2014]
            adj_lists = adj_all[year - 2014]
            feat_data = features_all[year - 2014]
            features = feat_data.clone().detach()

            # split pos neg sets for under-sampling
            train_pos, train_neg = pos_neg_split(idx_train, y_train)

            # # initialize model input
            # # features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
            # feat_data = normalize(feat_data)
            # features = torch.tensor(feat_data, dtype=torch.float32)
            # # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
            # if args.cuda:
            #     features.cuda()

            # # set input graph
            # if args.model == 'SAGE':
            #     adj_lists = homo
            # else:
            #     adj_lists = [relation1, relation2, relation3]

            # train the model
            # randomly under-sampling negative nodes for each epoch
            sampled_idx_train = undersample(train_pos, train_neg, scale=1)
            rd.shuffle(sampled_idx_train)

            # send number of batches to model to let the RLModule know the training progress
            num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
            if args.model == 'CARE':
                inter1.batch_num = num_batches

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
                        batch_nodes, Variable(torch.LongTensor(batch_label)),
                        adj_lists=adj_lists, features=features,
                        intra_list=[intra1, intra2, intra3])
                loss.backward()
                optimizer.step()
                end_time = time.time()
                epoch_time += end_time - start_time
                loss += loss.item()

            loss_sum += loss.item() / num_batches

        loss_average.append(loss_sum / (split_dicts[times]['train'][1] -
                                        split_dicts[times]['train'][0] + 1))

        # valid
        gnn_model.eval()

        a = test_care(
            range(len(y_all[split_dicts[times]['valid'] - 2014])),
            y_all[split_dicts[times]['valid'] - 2014], gnn_model,
            args.batch_size,
            features=features_all[split_dicts[times]['valid'] - 2014],
            adj_lists=adj_all[split_dicts[times]['valid'] - 2014],
            intra_list=[intra1, intra2, intra3], epoch=epoch, params=1)

        if epoch == 0:
            result = a
        else:
            result = pd.concat([result, a], axis=0)

        print("times ", times, " Epoch ", epoch, " Year ", year)

        a = test_care(range(len(y_all[split_dicts[times]['test'] - 2014])),
                      y_all[split_dicts[times]['test'] - 2014], gnn_model,
                      args.batch_size,
                      features=features_all[split_dicts[times]['test'] - 2014],
                      adj_lists=adj_all[split_dicts[times]['test'] - 2014],
                      intra_list=[intra1, intra2,
                                  intra3], epoch=epoch, params=2)
        result = pd.concat([result, a], axis=0)
    file_name = 'result/result_Findata_Nips_' + str(
        times) + '_' + 'Care_GNN' + '.csv'
    result.to_csv(file_name)

    # 将列表写入CSV文件
    file_name = 'result/result_Findata_Nips_train_loss_' + str(
        times) + '_' + 'Care_GNN' + '.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_loss_average'])
        for i, value in enumerate(loss_average):
            writer.writerow([i, value])
