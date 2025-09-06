"""This example implements the GIT-Mol model
(https://arxiv.org/abs/2308.06911) using PyG.
"""
import argparse
import os.path as osp

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.datasets import GitMolDataset
from torch_geometric.llm.models import GITMol
from torch_geometric.loader import DataLoader


@torch.no_grad()
def eval(model, data_loader):
    model.eval()
    loss = 0

    for batch in data_loader:
        batch_loss = model(batch.x, batch.edge_index, batch.batch,
                           batch.edge_attr, batch.smiles, batch.image,
                           batch.caption)
        loss += batch_loss.item() / len(data_loader)
    return loss


def train(
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    checkpointing: bool,
):
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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    model = GITMol().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare_data_loader(val_loader,
                                                 device_placement=True)
    test_loader = accelerator.prepare_data_loader(test_loader,
                                                  device_placement=True)

    # Train and eval ============================================
    best_epoch = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0
        if epoch == 0:
            print("Training beginning...")
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'

        for batch in tqdm(train_loader, desc=epoch_str):
            optimizer.zero_grad()
            loss = model(batch.x, batch.edge_index, batch.batch,
                         batch.edge_attr, batch.smiles, batch.image,
                         batch.caption)
            accelerator.backward(loss)

            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Eval
        val_loss = eval(model, val_loader)
        print(
            f'{epoch_str}, Train loss: {train_loss:4f}, Val loss: {val_loss:4f}'  # noqa: E501
        )

        if checkpointing and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    'model_state_dict':
                    accelerator.unwrap_model(model).state_dict(),
                    'best_loss':
                    best_val_loss
                },
                f'gitmol_pretrain_epoch{best_epoch}_val_loss{best_val_loss:4f}_ckpt.pt'  # noqa: E501
            )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Test
    test_loss = eval(model, test_loader)
    print(f'Test loss: {test_loss:4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--checkpointing', type=bool, default=True)
    args = parser.parse_args()

    train(
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.checkpointing,
    )
