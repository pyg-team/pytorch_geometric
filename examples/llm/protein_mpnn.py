"""This example implements the ProteinMPNN model
(https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1) using PyG.
"""
import argparse
import time

import numpy as np
import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import ProteinMPNNDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ProteinMPNN


def loss_smoothed(S, log_probs, mask, weight=0.1):
    # S: [batch_size]
    # log_probs: [batch_size, 21]
    # mask: [batch_size]
    """Negative log probabilities."""
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av


def loss_nll(S, log_probs, mask):
    """Negative log probabilities."""
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  # [B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def main(args):
    seed_everything(123)
    torch.amp.GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ProteinMPNNDataset(root=args.data_path, split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=4)

    model = ProteinMPNN(
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        num_positional_embedding=16,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5)

    total_step = 0

    for e in range(args.num_epochs):
        train_sum = 0.0
        train_acc = 0.0
        train_weights = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            # import pdb; pdb.set_trace()
            logits = model(batch.x, batch.edge_index, batch.edge_attr,
                           batch.chain_seq_label, batch.mask,
                           batch.chain_mask_all, batch.residue_idx,
                           batch.chain_encoding_all, batch.batch)
            # print(logits.size())
            mask_for_loss = batch.mask * batch.chain_mask_all
            y = batch.chain_seq_label
            _, loss = loss_smoothed(y, logits, mask_for_loss)
            loss.backward()

            optimizer.step()

            loss, loss_av, true_false = loss_nll(y, logits, mask_for_loss)

            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false *
                                   mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            # import pdb; pdb.set_trace()

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        # validation_loss = validation_sum / validation_weights
        # validation_accuracy = validation_acc / validation_weights
        # validation_perplexity = np.exp(validation_loss)

        train_perplexity_ = np.format_float_positional(
            np.float32(train_perplexity), unique=False, precision=3)
        validation_perplexity_ = 0  # np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)  # noqa: E501
        train_accuracy_ = np.format_float_positional(
            np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = 0  # np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3) # noqa: E501

        time.time()
        print(f'epoch: {e+1}, step: {total_step}, train: {train_perplexity_},'
              f'valid: {validation_perplexity_}, train_acc: {train_accuracy_},'
              f'valid_acc: {validation_accuracy_}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ProteinMPNN',
                        help='path for loading training data')
    # parser.add_argument('--path_for_outputs', type=str, default='./exp_020',
    #                     help='path for logs and model weights')
    # parser.add_argument('--previous_checkpoint', type=str, default='',
    #                     help='path for previous model weights, e.g. file.pt')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to train for')
    # parser.add_argument('--save_model_every_n_epochs', type=int, default=10,
    #                     help='save model weights every n epochs')
    # parser.add_argument('--reload_data_every_n_epochs', type=int, default=2,
    #                     help='reload training data every n epochs')
    # parser.add_argument(
    #     '--num_examples_per_epoch', type=int, default=1000000,
    #     help='number of training example to load for one epoch')
    # parser.add_argument('--batch_size', type=int, default=10000,
    #                     help='number of tokens for one batch')
    # parser.add_argument('--max_protein_length', type=int, default=10000,
    #                     help='maximum length of the protein complext')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='number of decoder layers')
    parser.add_argument('--num_neighbors', type=int, default=48,
                        help='number of neighbors for the sparse graph')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout level; 0.0 means no dropout')
    parser.add_argument(
        '--backbone_noise', type=float, default=0.2,
        help='amount of noise added to backbone during training')
    # parser.add_argument('--rescut', type=float, default=3.5,
    #                     help='PDB resolution cutoff')
    # parser.add_argument(
    #     '--gradient_norm', type=float, default=-1.0,
    #     help='clip gradient norm, set to negative to omit clipping')
    # parser.add_argument('--mixed_precision', type=bool, default=True,
    #                     help='train with mixed precision')

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print(f'Total Time: {time.time() - start_time:2f}s')
