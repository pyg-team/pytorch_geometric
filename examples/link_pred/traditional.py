import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAT, GCN, GraphSAGE
from torch_geometric.utils import negative_sampling, train_test_split_edges


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_i, z_j):
        x = torch.cat([z_i, z_j], dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.view(-1)


def train(encoder, predictor, data, optimizer):
    encoder.train()
    predictor.train()

    # Forward pass and calculate loss
    optimizer.zero_grad()

    # Get node embeddings
    z = encoder(data.x, data.train_pos_edge_index)

    # Positive edges
    pos_out = predictor(z[data.train_pos_edge_index[0]],
                        z[data.train_pos_edge_index[1]])

    # Sample and predict on negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )
    neg_out = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # Calculate loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_out,
                                                  torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out,
                                                  torch.zeros_like(neg_out))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(encoder, predictor, data):
    encoder.eval()
    predictor.eval()

    z = encoder(data.x, data.train_pos_edge_index)

    pos_val_out = predictor(z[data.val_pos_edge_index[0]],
                            z[data.val_pos_edge_index[1]])
    neg_val_out = predictor(z[data.val_neg_edge_index[0]],
                            z[data.val_neg_edge_index[1]])

    pos_test_out = predictor(z[data.test_pos_edge_index[0]],
                             z[data.test_pos_edge_index[1]])
    neg_test_out = predictor(z[data.test_neg_edge_index[0]],
                             z[data.test_neg_edge_index[1]])

    from sklearn.metrics import roc_auc_score

    val_pred = torch.cat([pos_val_out, neg_val_out]).sigmoid().cpu()
    val_true = torch.cat(
        [torch.ones(pos_val_out.size(0)),
         torch.zeros(neg_val_out.size(0))])
    val_auc = roc_auc_score(val_true, val_pred)

    test_pred = torch.cat([pos_test_out, neg_test_out]).sigmoid().cpu()
    test_true = torch.cat(
        [torch.ones(pos_test_out.size(0)),
         torch.zeros(neg_test_out.size(0))])
    test_auc = roc_auc_score(test_true, test_pred)

    return val_auc, test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSAGE'])
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                    args.dataset)
    dataset = Planetoid(path, args.dataset, transform=transform)
    data = dataset[0]

    # Process data for link prediction
    data = train_test_split_edges(data)

    # Initialize model
    if args.model == 'GCN':
        encoder = GCN(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model == 'GAT':
        encoder = GAT(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model == 'GraphSAGE':
        encoder = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
        ).to(device)

    # Initialize link predictor
    predictor = LinkPredictor(in_channels=args.out_channels,
                              hidden_channels=args.hidden_channels).to(device)

    print(f"Running traditional {args.model} on {args.dataset}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=args.lr)

    best_val_auc = 0
    final_test_auc = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(encoder, predictor, data, optimizer)
        val_auc, test_auc = test(encoder, predictor, data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, '
                f'Test AUC: {test_auc:.4f}')

    print(
        f'Final results - Val AUC: {best_val_auc:.4f}, Test AUC: {final_test_auc:.4f}'
    )


if __name__ == '__main__':
    main()
