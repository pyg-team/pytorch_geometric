import argparse
import os.path as osp

import torch
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.nn import SchNet
from torch_geometric.transforms import Compose, Distance, RadiusGraph

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
parser.add_argument(
    '--use_precomputed_edges', action='store_true',
    help='Compute graph edges and weights as a dataset pre_transform')
args = parser.parse_args()
pwd = osp.dirname(osp.realpath(__file__))

log(**vars(args))

if not args.use_precomputed_edges:
    path = osp.join(pwd, '..', 'data', 'QM9')
    dataset = QM9(path)
else:
    path = osp.join(pwd, '..', 'data', f'QM9_{args.cutoff}')
    pre_transform = Compose(
        [RadiusGraph(args.cutoff),
         Distance(norm=False, cat=False)])
    dataset = QM9(path, pre_transform=pre_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def default_predict(model, data):
    return model(data.z, data.pos, data.batch)


def precomputed_predict(model, data):
    return model(data.z, data.batch, data.edge_index, data.edge_attr)


if args.use_precomputed_edges:
    predict = precomputed_predict
else:
    predict = default_predict

for target in range(12):
    model, datasets = SchNet.from_qm9_pretrained(path, dataset, target)
    train_dataset, val_dataset, test_dataset = datasets

    model = model.to(device)
    loader = DataLoader(test_dataset, batch_size=256)

    maes = []
    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():
            pred = predict(model, data)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)

    mae = torch.cat(maes, dim=0)

    # Report meV instead of eV.
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
