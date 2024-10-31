"""This example implements the GIT-Mol model
(https://arxiv.org/abs/2308.06911) using PyG.
"""
import argparse
import os.path as osp
import time

import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import StepLR

from torch_geometric import seed_everything
from torch_geometric.datasets import GitMolDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GITMol


def train(
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    checkpointing: bool,
):
    time.time()
    # Load dataset ================================================
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', '..', 'data', 'GITMol')
    train_dataset = GitMolDataset(path, split=0)
    val_dataset = GitMolDataset(path, split=1)
    test_dataset = GitMolDataset(path, split=2)

    seed_everything(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Create model ===============================================
    accelerator = Accelerator()
    device = accelerator.device
    model = GITMol().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare_data_loader(val_loader,
                                                 device_placement=True)
    test_loader = accelerator.prepare_data_loader(test_loader,
                                                  device_placement=True)

    import pdb
    pdb.set_trace()
    # Train and eval ============================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--checkpointing', type=bool, default=True)
    args = parser.parse_args()

    start_time = time.time()
    train(
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.checkpointing,
    )
    print(f'Total Time: {time.time() - start_time:2f}s')
