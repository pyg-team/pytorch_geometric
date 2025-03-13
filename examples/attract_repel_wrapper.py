import argparse
import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import AttractRepel
from torch_geometric.utils import negative_sampling, train_test_split_edges


def train(model, data, optimizer):
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    model(data.x, data.train_pos_edge_index)

    # Link prediction on positive edges
    pos_out = model(data.x, data.train_pos_edge_index,
                    edge_index=data.train_pos_edge_index)

    # Sample negative edges and predict
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )

    neg_out = model(data.x, data.train_pos_edge_index,
                    edge_index=neg_edge_index)

    # Compute loss
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
    loss = pos_loss + neg_loss

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    # Get embeddings
    attract_z, repel_z = model.encode(data.x, data.train_pos_edge_index)

    # Link prediction on validation set
    pos_val_out = model.decode(attract_z, repel_z, data.val_pos_edge_index)
    neg_val_out = model.decode(attract_z, repel_z, data.val_neg_edge_index)

    # Link prediction on test set
    pos_test_out = model.decode(attract_z, repel_z, data.test_pos_edge_index)
    neg_test_out = model.decode(attract_z, repel_z, data.test_neg_edge_index)

    # Calculate AUC
    from sklearn.metrics import roc_auc_score

    val_preds = torch.cat(
        [torch.sigmoid(pos_val_out),
         torch.sigmoid(neg_val_out)])
    val_labels = torch.cat(
        [torch.ones_like(pos_val_out),
         torch.zeros_like(neg_val_out)])
    val_auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())

    test_preds = torch.cat(
        [torch.sigmoid(pos_test_out),
         torch.sigmoid(neg_test_out)])
    test_labels = torch.cat(
        [torch.ones_like(pos_test_out),
         torch.zeros_like(neg_test_out)])
    test_auc = roc_auc_score(test_labels.cpu(), test_preds.cpu())

    return val_auc, test_auc, model.calculate_r_fraction(attract_z, repel_z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--base_model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSAGE'])
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attract_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # Process data for link prediction
    data = train_test_split_edges(data)

    # Create AttractRepel model with specified base model
    model = AttractRepel(args.base_model, in_channels=dataset.num_features,
                         hidden_channels=args.hidden_channels,
                         out_channels=args.out_channels,
                         num_layers=args.num_layers,
                         attract_ratio=args.attract_ratio).to(device)

    print(
        f"Testing AttractRepel with {args.base_model} on {args.dataset} dataset"
    )
    print(
        f"Attract ratio: {args.attract_ratio}, Dimensions: {model.attract_dim} attract, {model.repel_dim} repel"
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_auc = 0
    best_test_auc = 0
    best_r_fraction = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        val_auc, test_auc, r_fraction = test(model, data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_r_fraction = r_fraction

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, '
                f'Test AUC: {test_auc:.4f}, R-fraction: {r_fraction:.4f}')

    print(f"Final results for {args.base_model} on {args.dataset}:")
    print(f"Test AUC: {best_test_auc:.4f}, R-fraction: {best_r_fraction:.4f}")


if __name__ == '__main__':
    main()
