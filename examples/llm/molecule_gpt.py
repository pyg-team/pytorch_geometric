"""This example implements the MoleculeGPT model
(https://ai4d3.github.io/papers/34.pdf) using PyG.
"""
import argparse
import math
import os.path as osp
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.datasets import InstructMolDataset, MoleculeGPTDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models import MoleculeGPT
from torch_geometric.nn.nlp import LLM, SentenceTransformer


def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)


@torch.no_grad()
def eval(model, data_loader):
    model.eval()
    loss = 0

    for batch in data_loader:
        batch_loss = model(batch.x, batch.edge_index, batch.batch,
                           batch.edge_attr, batch.smiles, batch.instruction,
                           batch.y)
        loss += batch_loss.item() / len(data_loader)
    return loss


def train(
    dataset_name: str,
    num_epochs: int,
    lr: float,
    batch_size: int,
    checkpointing: bool,
):
    def adjust_learning_rate(param_group, LR, epoch):
        # Decay the learning rate with half-cycle cosine after warmup
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lr = LR
        else:
            lr = min_lr + (LR - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                               (num_epochs - warmup_epochs)))
        param_group['lr'] = lr
        return lr

    def get_clippable_params(params):
        return [
            p for p in params
            if isinstance(p, torch.Tensor) and not hasattr(p, '_spec')
        ]

    start_time = time.time()
    # Load dataset ================================================
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', '..', 'data', dataset_name)
    if dataset_name == 'MoleculeGPT':
        dataset = MoleculeGPTDataset(path)
    elif dataset_name == 'InstructMol':
        dataset = InstructMolDataset(path)
    train_size, val_size = int(0.8 * len(dataset)), int(0.1 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    seed_everything(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Create model ===============================================
    llm = LLM(
        # model_name='lmsys/vicuna-7b-v1.5',
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        num_params=1,
        dtype=torch.bfloat16,
    )

    graph_encoder = GINEConv(
        nn=torch.nn.Sequential(
            torch.nn.Linear(6, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
        ),
        train_eps=True,
        edge_dim=4,
    )

    smiles_encoder = SentenceTransformer(
        model_name='DeepChem/ChemBERTa-77M-MTR',
        pooling_strategy='last_hidden_state',
    )

    model = MoleculeGPT(
        llm=llm,
        graph_encoder=graph_encoder,
        smiles_encoder=smiles_encoder,
    )

    # Train and eval ============================================
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05,
        },
    ], betas=(0.9, 0.95))
    grad_steps = 2

    best_epoch = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0
        if epoch == 0:
            print(f"Total Preparation Time: {time.time() - start_time:2f}s")
            start_time = time.time()
            print("Training beginning...")
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = model(batch.x, batch.edge_index, batch.batch,
                         batch.edge_attr, batch.smiles, batch.instruction,
                         batch.y)
            loss.backward()
            clip_grad_norm_(
                get_clippable_params(optimizer.param_groups[0]['params']), 0.1)

            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch)

            optimizer.step()
            epoch_loss += loss.detach().item()

            if (step + 1) % grad_steps == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(train_loader)

        # Eval
        val_loss = eval(model, val_loader)
        print(
            f'{epoch_str}, Train loss: {train_loss:4f}, Val loss: {val_loss:4f}'  # noqa: E501
        )

        if checkpointing and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_params_dict(
                model,
                f'moleculegpt_epoch{best_epoch}_val_loss{best_val_loss:4f}_ckpt.pt'  # noqa: E501
            )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Total Training Time: {time.time() - start_time:2f}s")
    # Test
    test_loss = eval(model, test_loader)
    print(f'Test loss: {test_loss:4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='MoleculeGPT',
                        choices=['MoleculeGPT', 'InstructMol'],
                        help='Support MoleculeGPT and InstructMol')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--checkpointing', type=bool, default=True)
    args = parser.parse_args()

    start_time = time.time()
    train(
        args.dataset_name,
        args.epochs,
        args.lr,
        args.batch_size,
        args.checkpointing,
    )
    print(f'Total Time: {time.time() - start_time:2f}s')
