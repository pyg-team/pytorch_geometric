import os.path as osp
import random
from datetime import datetime

import torch
from torch.utils.data import random_split

from torch_geometric.datasets import AmazonBook
from torch_geometric.nn import LightGCN


def prepare_data(device, path):
    # Load AmazonBook dataset
    dataset = AmazonBook(path)
    data = dataset[0].to(device)

    num_usr = data['user'].num_nodes
    num_itm = data['book'].num_nodes

    # edge_index from train set
    edges = data['user', 'book'].edge_index
    edges[1] = edges[1] + num_usr
    edge_index = torch.cat([edges, torch.flip(edges, [0])], 1)

    # Prepare val and test set
    test_edges = data['user', 'book'].edge_label_index
    val_set, test_set = random_split(test_edges.T, [0.3, 0.7])
    val_set = val_set[:].T
    test_set = test_set[:].T
    val_set = [
        val_set[:, val_set[0, :] == usr][1, :] for usr in range(num_usr)
    ]
    test_set = [
        test_set[:, test_set[0, :] == usr][1, :] for usr in range(num_usr)
    ]

    return num_usr, num_itm, edge_index, val_set, test_set


def batch_uniform_sampling(num_usr, num_itm, batch_size, edge_index):
    # Randomly sample users to prevent active users dominate the trand
    users = random.choices(range(num_usr), k=batch_size)

    # For each sampled user, randomly sample a pos_item and a neg_item
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
    """
    Mini-batch training procedure
    """
    n_batches = (edge_index.shape[1] // (2 * batch_size)) + 1
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


def test(model, num_usr, edge_index, truth_list, test_batch_size, k=20):
    """
    Test procedure
    """
    users = list(range(num_usr))
    model.eval()
    precision = 0.0
    recall = 0.0

    with torch.no_grad():
        embeds = model.get_embedding(edge_index)
        for i in range(0, len(users), test_batch_size):
            # Raw prediction for batch users
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
                truth = truth_list[u].tolist()
                ground_truth.append(truth)

            # TopK
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


def save_model(path, model, optimizer, precision, recall, epoch=None,
               hyperparams=None):
    """
    Save a trained model to disk.
    """
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    torch.save(
        {
            'timestamp': dt_string,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'precision': precision,
            'recall': recall,
            'hyperparams': hyperparams
        }, path)

    print(f"{path} saved at {dt_string}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'AmazonBook'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    config = {
        "embedding_dim": 64,
        "n_layers": 3,
        "LR": 0.001,
        "BATCH_SIZE": 2048,
    }

    # init data, model, optimizer
    num_usr, num_itm, edge_index, val_set, test_set = prepare_data(
        device, path)
    model = LightGCN(num_nodes=num_usr + num_itm,
                     embedding_dim=config["embedding_dim"],
                     num_layers=config["n_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])

    # train - val loop
    best_recall = 0.0
    print(f"Start training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(35):
        train_bpr_loss = train(model, optimizer, num_usr, num_itm, edge_index,
                               batch_size=config["BATCH_SIZE"])
        timer(epoch, 'train_bpr_loss: ' + str(train_bpr_loss.item()))

        val_precision, val_recall = test(model, num_usr, edge_index, val_set,
                                         test_batch_size=20000, k=20)
        timer(
            epoch, 'val_precision: ' + str(val_precision) + ' | val_recall: ' +
            str(val_recall))
        # save best model
        if val_recall > best_recall:
            best_recall = val_recall
            save_model(path + "/LightGCN_best.pt", model, optimizer,
                       val_precision, val_recall, epoch=epoch,
                       hyperparams=config)

    # load and test best model
    best_model = torch.load(path + "/LightGCN_best.pt")
    best_epoch = best_model['epoch']
    best_val_precision = best_model['precision']
    best_val_recall = best_model['recall']

    test_model = LightGCN(num_nodes=num_usr + num_itm,
                          embedding_dim=config["embedding_dim"],
                          num_layers=config["n_layers"]).to(device)
    test_model.load_state_dict(best_model['model_state_dict'])

    precision, recall = test(model, num_usr, edge_index, test_set,
                             test_batch_size=20000, k=20)
    print(f"Best epoch ({best_epoch}): "
          f"Val Precision@20: {best_val_precision:>7f}"
          f" | Recall@20: {best_val_recall:>7f}")
    print(f"Model performance: Test Precision@20: {precision:>7f}"
          f" | Recall@20: {recall:>7f}")


if __name__ == "__main__":
    main()
