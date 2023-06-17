import os.path as osp
import random
from datetime import datetime

import torch

from torch_geometric.datasets import AmazonBook
from torch_geometric.nn import LightGCN


def prepare_data(device):
    # Load AmazonBook dataset
    dataset = 'AmazonBook'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = AmazonBook(path)
    data = dataset[0].to(device)

    num_usr = data['user'].num_nodes
    num_itm = data['book'].num_nodes

    # Group pos_items list for each user in train set
    edges = data['user', 'book'].edge_index
    edges[1] = edges[1] + num_usr
    edge_index = torch.cat([edges, torch.flip(edges, [0])], 1)

    test_edge_index = data['user', 'book'].edge_label_index

    return num_usr, num_itm, edge_index, test_edge_index


def batch_uniform_sampling(num_usr, num_itm, batch_size, edge_index):
    # Randomly sample users to prevent high activate user dominate the trand
    users = random.choices(range(num_usr), k=batch_size)

    # for each sampled user, randomly sample a pos_item and a neg_item
    u_p_n = []
    for user in users:
        pos_list = edge_index[:, edge_index[0, :] == user][1]
        if len(pos_list) == 0:
            continue
        pos_item = random.choice(pos_list.tolist())
        while True:
            neg_item = random.choice(range(num_usr, num_usr + num_itm))
            if neg_item not in pos_list:
                break
        u_p_n.append([user, pos_item, neg_item])
    u_p_n = torch.LongTensor(u_p_n).T
    return u_p_n[0], u_p_n[1], u_p_n[2]


def train(model, optimizer, num_usr, num_itm, edge_index, batch_size):
    '''
    Mini-batch training procedure
    '''
    n_batches = (edge_index.shape[1] // (2 * batch_size)) + 1
    n_batches = n_batches // 10
    avg_loss = 0
    model.train()
    for batch_idx in range(n_batches):
        optimizer.zero_grad()
        users, pos_items, neg_items = \
            batch_uniform_sampling(num_usr, num_itm, batch_size,
                                   edge_index)
        batch_edge_labels = torch.stack(
            (torch.cat([users, users]), torch.cat([pos_items, neg_items])))
        rankings = model(edge_index, batch_edge_labels)
        pos_rank, neg_rank = rankings.chunk(2)
        node_indices = torch.cat([users, pos_items, neg_items]).unique()
        bpr_loss = model.recommendation_loss(pos_rank, neg_rank, node_indices)
        bpr_loss.backward()
        optimizer.step()
        avg_loss += bpr_loss
    return avg_loss / n_batches


def test(model, num_usr, edge_index, test_edge_index, test_batch_size, k=20):
    '''
    Test procedure
    '''
    n_batches = 0
    users = list(range(num_usr))
    model.eval()
    precision = 0.0
    recall = 0.0

    with torch.no_grad():
        embeds = model.get_embedding(edge_index)
        for i in range(0, len(users), test_batch_size):
            n_batches += 1
            batch_users = users[i:i + test_batch_size]
            src = embeds[batch_users]
            dst = embeds[num_usr:]
            pred = src @ dst.t()

            exc_usr = []
            exc_itm = []
            ground_truth = []
            for key, u in enumerate(batch_users):
                # exclude users, items indices
                items = edge_index[:, edge_index[0, :] == u][1] - num_usr
                exc_usr.extend([key] * len(items))
                exc_itm.extend(items.tolist())
                # ground truth for batch users
                truth = test_edge_index[:, test_edge_index[0, :] ==
                                        u][1].tolist()
                ground_truth.append(truth)

            pred[exc_usr, exc_itm] = -(1 << 10)
            top_index = pred.topk(k, dim=-1).indices

            # sum precision and recall of each user in the batch
            skip = 0
            for usr in range(len(batch_users)):
                if len(ground_truth[usr]) != 0:
                    hit = sum(
                        list(
                            map(lambda x: x in ground_truth[usr],
                                top_index[usr])))
                    precision += hit / k
                    recall += hit / len(ground_truth[usr])
                else:
                    skip += 1
    # return average precision and recall cross all the users
    return precision / (len(users) - skip), recall / (len(users) - skip)


def timer(epoch, message):
    now = datetime.now().strftime('%H:%M:%S')
    print(f'{now} EPOCH[{epoch + 1}/{50}] {message}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_usr, num_itm, edge_index, test_edge_index = prepare_data(device)
    model = LightGCN(num_nodes=num_usr + num_itm, embedding_dim=60,
                     num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Start training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    avg_precision, avg_recall = test(model, num_usr, edge_index,
                                     test_edge_index, test_batch_size=10000,
                                     k=20)
    timer(
        -1, 'avg_precision: ' + str(avg_precision) + ' | avg_recall: ' +
        str(avg_recall))

    for epoch in range(50):
        avg_bpr_loss = train(model, optimizer, num_usr, num_itm, edge_index,
                             batch_size=2048)
        timer(epoch, 'avg_bpr_loss: ' + str(avg_bpr_loss.item()))
        if epoch % 1 == 0:
            avg_precision, avg_recall = test(model, num_usr, edge_index,
                                             test_edge_index,
                                             test_batch_size=20000, k=20)
            timer(
                epoch, 'avg_precision: ' + str(avg_precision) +
                ' | avg_recall: ' + str(avg_recall))


if __name__ == "__main__":
    main()
