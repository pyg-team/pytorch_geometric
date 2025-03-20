import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models.attract_repel import AttractRepel
from torch_geometric.utils import negative_sampling, train_test_split_edges


def train(model, data, optimizer):
    model.train()

    # Forward pass and calculate loss
    optimizer.zero_grad()

    # Generate prediction scores for positive edges
    pos_out = model(data.x, data.train_pos_edge_index,
                    edge_label_index=data.train_pos_edge_index)

    # Sample and predict on negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )

    neg_out = model(data.x, data.train_pos_edge_index,
                    edge_label_index=neg_edge_index)

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
def test(model, data):
    model.eval()

    # Evaluate validation and test edges
    pos_val_out = model(data.x, data.train_pos_edge_index,
                        edge_label_index=data.val_pos_edge_index)

    neg_val_out = model(data.x, data.train_pos_edge_index,
                        edge_label_index=data.val_neg_edge_index)

    pos_test_out = model(data.x, data.train_pos_edge_index,
                         edge_label_index=data.test_pos_edge_index)

    neg_test_out = model(data.x, data.train_pos_edge_index,
                         edge_label_index=data.test_neg_edge_index)

    # Calculate AUC scores
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
    parser.add_argument('--attract_ratio', type=float, default=0.5,
                        help='Proportion of dimensions for attract component')
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

    # Initialize model using the wrapper
    model = AttractRepel(
        model=args.model,
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        attract_ratio=args.attract_ratio,
    ).to(device)

    print(
        f"Running {args.model} with Attract-Repel embeddings on {args.dataset}"
    )
    print(
        f"Attract dimensions: {model.attract_dim}, Repel dimensions: {model.repel_dim}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = 0
    final_test_auc = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        val_auc, test_auc = test(model, data)

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

    # Calculate and report R-fraction
    with torch.no_grad():
        attract_z, repel_z = model.encode(data.x, data.train_pos_edge_index)
        r_fraction = model.calculate_r_fraction(attract_z, repel_z)
        print(f"R-fraction: {r_fraction.item():.4f}")


if __name__ == '__main__':
    main()
