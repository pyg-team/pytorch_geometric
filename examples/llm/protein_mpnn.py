"""This example implements the ProteinMPNN model
(https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1) using PyG.
"""
import argparse
import time

import torch

from torch_geometric.datasets import ProteinMPNNDataset
from torch_geometric.loader import DataLoader


def main(args):
    torch.amp.GradScaler()
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ProteinMPNNDataset(root=args.data_path, split='train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False,
                              num_workers=6)

    for batch in train_loader:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ProteinMPNN',
                        help='path for loading training data')
    parser.add_argument('--path_for_outputs', type=str, default='./exp_020',
                        help='path for logs and model weights')
    parser.add_argument('--previous_checkpoint', type=str, default='',
                        help='path for previous model weights, e.g. file.pt')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('--save_model_every_n_epochs', type=int, default=10,
                        help='save model weights every n epochs')
    parser.add_argument('--reload_data_every_n_epochs', type=int, default=2,
                        help='reload training data every n epochs')
    parser.add_argument(
        '--num_examples_per_epoch', type=int, default=1000000,
        help='number of training example to load for one epoch')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='number of tokens for one batch')
    parser.add_argument('--max_protein_length', type=int, default=10000,
                        help='maximum length of the protein complext')
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
    parser.add_argument('--rescut', type=float, default=3.5,
                        help='PDB resolution cutoff')
    parser.add_argument('--debug', type=bool, default=False,
                        help='minimal data loading for debugging')
    parser.add_argument(
        '--gradient_norm', type=float, default=-1.0,
        help='clip gradient norm, set to negative to omit clipping')
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help='train with mixed precision')

    args = parser.parse_args()

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    start_time = time.time()
    main(args)
    print(f'Total Time: {time.time() - start_time:2f}s')
