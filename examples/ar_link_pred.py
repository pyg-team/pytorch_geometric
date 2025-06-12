import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


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


class ARLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Split dimensions between attract and repel
        self.attract_dim = in_channels // 2
        self.repel_dim = in_channels - self.attract_dim

    def forward(self, z_i, z_j):
        # Split into attract and repel parts
        z_i_attr = z_i[:, :self.attract_dim]
        z_i_repel = z_i[:, self.attract_dim:]

        z_j_attr = z_j[:, :self.attract_dim]
        z_j_repel = z_j[:, self.attract_dim:]

        # Calculate AR score
        attract_score = (z_i_attr * z_j_attr).sum(dim=1)
        repel_score = (z_i_repel * z_j_repel).sum(dim=1)

        return attract_score - repel_score


def train(encoder, predictor, data, optimizer):
    encoder.train()
    predictor.train()

    # Forward pass and calculate loss
    optimizer.zero_grad()
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

    val_auc = compute_auc(pos_val_out, neg_val_out)
    test_auc = compute_auc(pos_test_out, neg_test_out)

    return val_auc, test_auc


def compute_auc(pos_out, neg_out):
    pos_out = torch.sigmoid(pos_out).cpu().numpy()
    neg_out = torch.sigmoid(neg_out).cpu().numpy()

    # Simple AUC calculation
    from sklearn.metrics import roc_auc_score
    y_true = torch.cat(
        [torch.ones(pos_out.shape[0]),
         torch.zeros(neg_out.shape[0])])
    y_score = torch.cat([torch.tensor(pos_out), torch.tensor(neg_out)])

    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_ar', action='store_true',
                        help='Use Attract-Repel embeddings')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)
    dataset = Planetoid(path, args.dataset, transform=transform)
    data = dataset[0]

    # Process data for link prediction
    data = train_test_split_edges(data)

    # Initialize encoder
    encoder = GCNEncoder(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
    ).to(device)

    # Choose predictor based on args
    if args.use_ar:
        predictor = ARLinkPredictor(in_channels=args.out_channels).to(device)
        print(f"Running link prediction on {args.dataset}"
              f"with Attract-Repel embeddings")
    else:
        predictor = LinkPredictor(
            in_channels=args.out_channels,
            hidden_channels=args.hidden_channels).to(device)
        print(f"Running link prediction on {args.dataset}"
              f"with Traditional embeddings")

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
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Val AUC: {val_auc:.4f}, '
                  f'Test AUC: {test_auc:.4f}')

    print(f'Final results - Val AUC: {best_val_auc:.4f}, '
          f'Test AUC: {final_test_auc:.4f}')

    # Calculate R-fraction if using AR
    if args.use_ar:
        with torch.no_grad():
            z = encoder(data.x, data.train_pos_edge_index)
            attr_dim = args.out_channels // 2

            z_attr = z[:, :attr_dim]
            z_repel = z[:, attr_dim:]

            attract_norm_squared = torch.sum(z_attr**2)
            repel_norm_squared = torch.sum(z_repel**2)

            r_fraction = repel_norm_squared / (attract_norm_squared +
                                               repel_norm_squared)
            print(f"R-fraction: {r_fraction.item():.4f}")


if __name__ == '__main__':
    main()
