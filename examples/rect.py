import argparse
import os.path as osp
from sklearn.linear_model import LogisticRegression
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric_test.transforms as T
from torch_geometric_test.nn.models import RECT_L
from torch_geometric_test.nn.models.rect_net import get_semantic_labels


def evaluate_embeds(embeddings, labels, train_mask, test_mask):
    '''Evaluate via training a logistic regression classifier
    with the original train/test split setting '''
    model = LogisticRegression()
    model.fit(embeddings[train_mask].cpu().numpy(),
              labels[train_mask].cpu().numpy())
    return model.score(embeddings[test_mask].cpu(), labels[test_mask].cpu())


def run(args, data, original):
    '''As RECT focus on the zero-shot (i.e., completely-imbalanced) label setting,
    we first remove "unseen" classes from the training set.
       Then, we train a RECT (or more specifically its supervised part RECT-L)
    model with the zero shot labels.
       Finally, with the embeddings and original labels,
    we train a classifier to evaluate the performance.'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model, features = RECT_L(in_feats=args.n_hidden,
                             n_hidden=args.n_hidden,
                             dropout=0.0).to(device), data.x
    semantic_labels = get_semantic_labels(train_mask_zs=data.train_mask,
                                          labels=data.y,
                                          features=features).to(device)
    loss_fcn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001, weight_decay=5e-4)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.edge_index, data.edge_attr)
        loss_train = loss_fcn(semantic_labels, logits[data.train_mask])
        print('Epoch {:d} | Train Loss {:.5f}'.format(epoch + 1,
                                                      loss_train.item()))
        loss_train.backward()
        optimizer.step()
    model.eval()

    res_rect = evaluate_embeds(embeddings=model.embed(features), labels=data.y,
                               train_mask=train_mask_original,
                               test_mask=data.test_mask)
    print("Test Accuracy of {:s}: {:.4f}".format('RECT-L', res_rect))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'], help="dataset")
    parser.add_argument("--removed-classes", type=int, nargs='*',
                        default={1, 2, 3}, help="unseen classes")
    parser.add_argument("--n-hidden", type=int,
                        default=200, help="number of hidden units")
    args = parser.parse_args()
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', args.dataset)
    train_mask_original = Planetoid(path, args.dataset)[0].train_mask.clone()
    transform = T.Compose([T.NormalizeFeatures(),
                           T.GetzsTrainMask(args.removed_classes),
                           T.FeatureReduction(args.n_hidden)])
    dataset = Planetoid(path, args.dataset, transform=transform)
    data = T.GDC()(dataset[0])
    print('after removing the unseen classes, seen class labeled node num:',
          sum(data.train_mask).item())
    run(args, data, train_mask_original)
