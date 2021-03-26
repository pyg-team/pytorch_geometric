# match same input of features
#from Datas import dataloader
import pandas as pd
target_list = ['Result0']
import pickle

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from rdkit.Chem import rdMolTransforms, MolFromSmiles
from torch_geometric.data import Data
import random
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import pandas as pd


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# return 40 atom feature added "6" in atom degree (yeh it can happend!)
def atom_features(atom, explicit_H=False, use_chirality=True):

    results = one_of_k_encoding_unk(atom.GetSymbol(),
      [ 'B', 'C', 'N', 'O','F', 'Si', 'P','S','Cl','As','Se','Br', 'Te','I',  'At','other']) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
          [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
          one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
          ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def mol2vec(mol, use_chirality=True):
    """ Main function that is called to create the features for the dataloader """
    full_atom_feature_dims = []
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    mydtype = torch.float

    # get atom feature labels
    node_features = [
        atom_features(atom, use_chirality=use_chirality) for atom in atoms
    ]

    # get adjacency matrix to egde_index
    edge_index = torch.tensor(get_bond_pair(mol), dtype=torch.long)

    # get bond feature labels
    edge_features = [bond_features(bond) for bond in bonds]
    # caution: for bigraph molecules (ie undirected graphs) need to inject twice the bond features
    for bond in bonds:
        edge_features.append(bond_features(bond))

    # format python vectors to torch vectors
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    x = torch.tensor(node_features, dtype=mydtype)
    # convert to pytorch Data storage
    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


def datagenerator(df, target_list, use_chirality=True):
    # basic single task version for testing
    # extract the pandas and extract the target_list and smiles column
    # smiles is converted to rkdit mol object and than
    # rdkit mol object is converted to mol pytorch graph collection
    # finally the collection is populate with target per molecule.
    # output: a Data collecion containing input and output (graph, and Y target)
    nbtarget = len(target_list)
    train_list = []
    smiles = df['smiles']

    for i, smi in enumerate(tqdm(smiles)):
        y = df[target_list].loc[i]

        mol = Chem.MolFromSmiles(smi)

        train_X = [mol2vec(mol, use_chirality=use_chirality)]

        for data in train_X:
            data['y'] = torch.tensor([np.array(y)], dtype=torch.float)
        train_list.extend(train_X)

    return train_list


def getdataloader(datacollection, batch_size=32, shuffle=False,
                  drop_last=False):
    # convert the molecules Data colection in Batch data
    train_loader = DataLoader(datacollection, batch_size=batch_size,
                              shuffle=shuffle, drop_last=drop_last)
    return train_loader


df = pd.read_csv('../alpha/esol.csv')
print(df)
data = datagenerator(df, target_list)

batch_size = 128

train_loader = getdataloader(data, batch_size, shuffle=True, drop_last=False)

for data in train_loader:
    print(data)
    print(data.x.shape)
    break

from torch_geometric.nn.models.attentive_fp import AttentiveFP
# generate the model architecture
model = AttentiveFP(in_channels=40, hidden_channels=200, out_channels=1,
                    edge_dim=10, num_layers=2, dropout=0.2)

# loop over data in a batch
import time
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

# train the network
for t in range(200):
    start = time.time()
    epoch_loss = 0
    count_train = 0
    for data in train_loader:
        y = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = loss_func(y.squeeze(),
                         data.y.squeeze())  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        count_train += data.y.size(0)

        with torch.no_grad():
            epoch_loss += data.num_graphs * loss.item()

        stop = time.time()

        print('epoch', t + 1, ' - train loss:', epoch_loss / count_train,
              ' - runtime:', stop - start)
