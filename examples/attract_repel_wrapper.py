import argparse
import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import AttractRepel
from torch_geometric.utils import negative_sampling, train_test_split_edges


def train(model, data, optimizer, is_ar=True):
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    if is_ar:
        z = model(data.x, data.train_pos_edge_index)

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
    else:
        # For base models, use a simple dot product for link prediction
        z = model(data.x, data.train_pos_edge_index)

        # Link prediction on positive edges
        pos_out = (z[data.train_pos_edge_index[0]] *
                   z[data.train_pos_edge_index[1]]).sum(dim=1)

        # Sample negative edges and predict
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
        )

        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # Compute loss
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
    loss = pos_loss + neg_loss

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, is_ar=True):
    model.eval()

    if is_ar:
        # Get embeddings for attract-repel model
        attract_z, repel_z = model.encode(data.x, data.train_pos_edge_index)

        # Link prediction on validation set
        pos_val_out = model.decode(attract_z, repel_z, data.val_pos_edge_index)
        neg_val_out = model.decode(attract_z, repel_z, data.val_neg_edge_index)

        # Link prediction on test set
        pos_test_out = model.decode(attract_z, repel_z,
                                    data.test_pos_edge_index)
        neg_test_out = model.decode(attract_z, repel_z,
                                    data.test_neg_edge_index)

        r_fraction = model.calculate_r_fraction(attract_z, repel_z)
    else:
        # For base models, get embeddings and use dot product
        z = model(data.x, data.train_pos_edge_index)

        # Link prediction using dot product
        pos_val_out = (z[data.val_pos_edge_index[0]] *
                       z[data.val_pos_edge_index[1]]).sum(dim=1)
        neg_val_out = (z[data.val_neg_edge_index[0]] *
                       z[data.val_neg_edge_index[1]]).sum(dim=1)

        pos_test_out = (z[data.test_pos_edge_index[0]] *
                        z[data.test_pos_edge_index[1]]).sum(dim=1)
        neg_test_out = (z[data.test_neg_edge_index[0]] *
                        z[data.test_neg_edge_index[1]]).sum(dim=1)

        r_fraction = 0.0  # No R-fraction for base models

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

    return val_auc, test_auc, r_fraction


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
    parser.add_argument('--no_ar', action='store_true',
                        help='Use base model without attract-repel')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # Process data for link prediction
    data = train_test_split_edges(data)

    if args.no_ar:
        # Import base models
        from torch_geometric.nn.models.basic_gnn import GAT, GCN, GraphSAGE

        # Create base model directly
        if args.base_model == 'GCN':
            model = GCN(in_channels=dataset.num_features,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.out_channels,
                        num_layers=args.num_layers).to(device)
        elif args.base_model == 'GAT':
            model = GAT(in_channels=dataset.num_features,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.out_channels,
                        num_layers=args.num_layers).to(device)
        elif args.base_model == 'GraphSAGE':
            model = GraphSAGE(in_channels=dataset.num_features,
                              hidden_channels=args.hidden_channels,
                              out_channels=args.out_channels,
                              num_layers=args.num_layers).to(device)
        print(
            f"Testing base {args.base_model} model on {args.dataset} dataset")
    else:
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
        loss = train(model, data, optimizer, is_ar=not args.no_ar)
        val_auc, test_auc, r_fraction = test(model, data, is_ar=not args.no_ar)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_r_fraction = r_fraction

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, '
                f'Test AUC: {test_auc:.4f}, R-fraction: {r_fraction:.4f}')

    if args.no_ar:
        print(f"Final results for base {args.base_model} on {args.dataset}:")
        print(f"Test AUC: {best_test_auc:.4f}")
    else:
        print(
            f"Final results for AttractRepel with {args.base_model} on {args.dataset}:"
        )
        print(
            f"Test AUC: {best_test_auc:.4f}, R-fraction: {best_r_fraction:.4f}"
        )


if __name__ == '__main__':
    main()
