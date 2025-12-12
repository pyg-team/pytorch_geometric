import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import WebQSPDatasetReaRev
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn.models import ReaRev


def get_loss(model, batch):
    """Get batch normalized KL divergence loss"""
    probs = model(batch.question_tokens, batch.question_mask, batch.x,
                  batch.edge_index, batch.edge_type, batch.edge_attr,
                  batch.seed_mask, batch.batch).view(-1)

    log_probs = torch.log(torch.clamp(probs, min=1e-10))
    y = batch.y.view(-1)

    num_graphs = batch.num_graphs
    ans_per_graph = torch.zeros(num_graphs, device=probs.device)
    ans_per_graph = ans_per_graph.scatter_add_(0, batch.batch, y)
    denom = ans_per_graph.clamp(min=1.0)
    teacher = y / denom[batch.batch]

    # Using reduction="batchmean" directly will divide by all nodes, so we manually normalize by batch
    kl_per_node = F.kl_div(log_probs, teacher, reduction="none")
    kl_per_graph = torch.zeros(num_graphs, device=probs.device)
    kl_per_graph = kl_per_graph.scatter_add_(0, batch.batch, kl_per_node)

    valid_mask = (ans_per_graph > 0).float()
    valid_count = valid_mask.sum().clamp(min=1)

    return (kl_per_graph * valid_mask).sum() / valid_count


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = get_loss(model, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, loader, device):
    """Compute Hit@1 average"""
    model.eval()
    hits1 = 0
    total_graphs = 0

    for batch in loader:

        batch = batch.to(device)
        probs = model(
            batch.question_tokens,
            batch.question_mask,
            batch.x,
            batch.edge_index,
            batch.edge_type,
            batch.edge_attr,
            batch.seed_mask,
            batch.batch,
        )

        batch_ptr = batch.ptr
        current_num_graphs = batch.num_graphs
        total_graphs += current_num_graphs
        for i in range(current_num_graphs):
            start = batch_ptr[i]
            end = batch_ptr[i + 1]
            graph_probs = probs[start:end].view(-1)
            graph_targets = batch.y[start:end].view(-1)
            pred_idx = graph_probs.argmax().item()

            if graph_targets[pred_idx] > 0:
                hits1 += 1

    return hits1 / total_graphs


def main(root, batch_size, epochs, num_layers, num_instructions,
         num_iterations, save_path):
    HIDDEN_CHANNELS = 128
    QUESTION_EMBED_DIM = 384
    EDGE_ATTR_DIM = 384
    NODE_IN_CHANNELS = 384
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WebQSPDatasetReaRev(root=root, split="train")
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = WebQSPDatasetReaRev(root=root, split="validation")
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False)

    model = ReaRev(
        NODE_IN_CHANNELS,
        EDGE_ATTR_DIM,
        HIDDEN_CHANNELS,
        num_layers,
        num_instructions,
        num_iterations,
        QUESTION_EMBED_DIM,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    print("Training...")
    for epoch in range(epochs):
        avg_loss = train(model, loader, optimizer, device)
        print(
            f"Epoch {epoch+1} Training Loss: {avg_loss:.4f} Val Hit@1: {test(model, val_loader, device):.4f}"
        )

    print("Saving model...")
    config = {
        'node_in_channels': NODE_IN_CHANNELS,
        'edge_attr_dim': EDGE_ATTR_DIM,
        'hidden_channels': HIDDEN_CHANNELS,
        'num_layers': num_layers,
        'num_instructions': num_instructions,
        'num_iterations': num_iterations,
        'question_embed_dim': QUESTION_EMBED_DIM,
    }
    checkpoint = {'config': config, 'state_dict': model.state_dict()}
    torch.save(checkpoint, save_path)
    print("Model saved to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ReaRev on WebQSP.")
    parser.add_argument("--root", type=str, default="./data/webqsp",
                        help="Dataset root directory.")
    parser.add_argument("--save_path", type=str,
                        default="../outputs/rearev/rearev.pth",
                        help="Path to save the model.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of reasoning layers.")
    parser.add_argument("--num_instructions", type=int, default=2,
                        help="Number of instructions.")
    parser.add_argument("--num_iterations", type=int, default=2,
                        help="Number of adaptation iterations.")
    args = parser.parse_args()

    main(
        root=args.root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_layers=args.num_layers,
        num_instructions=args.num_instructions,
        num_iterations=args.num_iterations,
        save_path=args.save_path,
    )
