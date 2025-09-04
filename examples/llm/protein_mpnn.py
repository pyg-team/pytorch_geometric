"""This example implements the ProteinMPNN model
(https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1) using PyG.
"""
import argparse
import time

import numpy as np
import psutil
import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import ProteinMPNNDataset
from torch_geometric.llm.models import ProteinMPNN
from torch_geometric.loader import DataLoader


def loss_smoothed(y, logits, mask, weight=0.1):
    """Negative log probabilities."""
    y_onehot = torch.nn.functional.one_hot(y, 21).float()

    # Label smoothing
    y_onehot = y_onehot + weight / float(y_onehot.size(-1))
    y_onehot = y_onehot / y_onehot.sum(-1, keepdim=True)

    loss = -(y_onehot * logits).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av


def loss_nll(y, logits, mask):
    """Negative log probabilities."""
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(logits.contiguous().view(-1, logits.size(-1)),
                     y.contiguous().view(-1)).view(y.size())
    y_argmaxed = torch.argmax(logits, -1)  # [B, L]
    true_false = (y == y_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement learning rate above."""
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5) *
                              min(step**(-0.5), step * self.warmup**(-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def train(model, optimizer, data_loader, device, scaler):
    model.train()
    train_sum = 0.0
    train_acc = 0.0
    train_weights = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        mask_for_loss = batch.mask * batch.chain_mask_all
        y = batch.chain_seq_label

        if torch.cuda.is_available() and args.mixed_precision:
            with torch.amp.autocast('cuda'):
                logits = model(batch.x, batch.chain_seq_label, batch.mask,
                               batch.chain_mask_all, batch.residue_idx,
                               batch.chain_encoding_all, batch.batch)
                _, loss = loss_smoothed(y, logits, mask_for_loss)

            scaler.scale(loss).backward()

            if args.gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.gradient_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch.x, batch.chain_seq_label, batch.mask,
                           batch.chain_mask_all, batch.residue_idx,
                           batch.chain_encoding_all, batch.batch)

            _, loss = loss_smoothed(y, logits, mask_for_loss)
            loss.backward()

            if args.gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.gradient_norm)

            optimizer.step()

        loss, _, true_false = loss_nll(y, logits, mask_for_loss)

        train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_accuracy = train_acc / train_weights
    train_perplexity = np.exp(train_loss)

    return train_perplexity, train_accuracy


@torch.no_grad()
def eval(model, data_loader, device):
    model.eval()
    valid_sum = 0.
    valid_weights = 0.
    valid_acc = 0.
    for batch in data_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.chain_seq_label, batch.mask,
                       batch.chain_mask_all, batch.residue_idx,
                       batch.chain_encoding_all, batch.batch)

        mask_for_loss = batch.mask * batch.chain_mask_all
        y = batch.chain_seq_label
        loss, _, true_false = loss_nll(y, logits, mask_for_loss)

        valid_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        valid_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        valid_weights += torch.sum(mask_for_loss).cpu().data.numpy()

    valid_loss = valid_sum / valid_weights
    valid_accuracy = valid_acc / valid_weights
    valid_perplexity = np.exp(valid_loss)

    return valid_perplexity, valid_accuracy


def main(args):
    wall_clock_start = time.perf_counter()
    seed_everything(123)
    scaler = torch.amp.GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.size == 'large' and psutil.virtual_memory().total < 64.1 * 1024**3:
        print('Warning: may not have enough RAM to run this example.')
        print('Consider upgrading RAM if an error occurs.')
        print('Estimated RAM Needed: ~64.1GB.')

    train_dataset = ProteinMPNNDataset(
        root=args.data_path,
        size=args.size,
        split='train',
        rescut=args.rescut,
        max_length=args.max_protein_length,
    )
    valid_dataset = ProteinMPNNDataset(
        root=args.data_path,
        size=args.size,
        split='valid',
        rescut=args.rescut,
        max_length=args.max_protein_length,
    )
    test_dataset = ProteinMPNNDataset(
        root=args.data_path,
        size=args.size,
        split='test',
        rescut=args.rescut,
        max_length=args.max_protein_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                              shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False, num_workers=6)

    model = ProteinMPNN(
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        num_positional_embedding=16,
    ).to(device)

    total_step = 0
    optimizer = NoamOpt(
        model_size=args.hidden_dim, factor=2, warmup=4000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                   eps=1e-9), step=total_step)

    times = []
    for e in range(args.num_epochs):
        start = time.perf_counter()
        train_perplexity, train_accuracy = train(model, optimizer,
                                                 train_loader, device, scaler)
        valid_perplexity, valid_accuracy = eval(model, valid_loader, device)

        print(
            f'epoch: {e:03d}, step: {total_step}, '
            f'train: {train_perplexity:.3f}, valid: {valid_perplexity:.3f}, '
            f'train_acc: {train_accuracy:.3f}, valid_acc: {valid_accuracy:.3f}'
        )
        times.append(time.perf_counter() - start)

    print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
    print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
    print(f'Total Program Runtime: '
          f'{time.perf_counter() - wall_clock_start:.4f}s')
    # Test
    test_perplexity, test_accuracy = eval(model, test_loader, device)
    print(f'test: {test_perplexity:.3f}, test_acc: {test_accuracy:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument('--data_path', type=str, default='data/ProteinMPNN',
                        help='path for loading training data')
    parser.add_argument(
        '--size', type=str, default='small', choices=['small', 'large'],
        help='Use of "small (229.4 MB)" or "large (64.1 GB)" dataset')
    parser.add_argument('--max_protein_length', type=int, default=10000,
                        help='maximum length of the protein complext')
    parser.add_argument('--rescut', type=float, default=3.5,
                        help='PDB resolution cutoff')
    # training config
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs to train for')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='number of tokens for one train batch')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='number of tokens for one valid or test batch')
    parser.add_argument(
        '--gradient_norm', type=float, default=-1.0,
        help='clip gradient norm, set to negative to omit clipping')
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help='train with mixed precision')
    # model config
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='number of decoder layers')
    parser.add_argument('--num_neighbors', type=int, default=30,
                        help='number of neighbors for the sparse graph')
    parser.add_argument('--num_rbf', type=int, default=16,
                        help='number of radial basis functions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout level; 0.0 means no dropout')
    parser.add_argument(
        '--backbone_noise', type=float, default=0.2,
        help='amount of noise added to backbone during training')

    args = parser.parse_args()

    main(args)
