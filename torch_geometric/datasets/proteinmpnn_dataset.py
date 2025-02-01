import csv
import os
import os.path as osp
import random

import numpy as np
import torch
from dateutil import parser

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.io import fs


class PMPNNDataset(InMemoryDataset):
    r"""Dataset used in the training example from the `"Robust deep learning based
    protein sequence design using ProteinMPNN"
    <https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1.full.pdf>'_paper.

    Args:
      root (str): Root directory where the dataset should be saved.
      params (dict): Dictionary of parameters for dataset creation:
                     LIST: Path to the table with metadata.
                     VAL: Path to list of cluster IDs for model validation.
                     DIR: Path to dataset.
                     DATCUT: Date (YYY-MM-DD) threshold of sequence deposition.
                     RESCUT: Resolution cutoff for PDBs.
                     HOMO: Minimal sequence identity to detect homodimeric chains.
      set_type (str): Type of expected data, train ("train") or validation ("val")
    """

    url = 'https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz'
    dir_name = 'pdb_2021aug02_sample'

    def __init__(
            self,
            root,
            set_type,  # 'train', 'val', or 'test'
            params,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            log=True,
            force_reload=False
        #name='sample',
    ) -> None:
        assert set_type in {'train', 'val'}
        self.params = params
        self.set_type = set_type
        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)
        path = self.processed_paths[0]
        self.load(path)

    @property
    def raw_file_names(self):
        return [self.dir_name + '.tar.gz']

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt']

    @property
    def raw_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f)
                for f in files] or osp.join(self.raw_dir, files)

    def download(self):
        path = download_url(self.url, self.root)
        extract_tar(path, self.root)
        os.unlink(path)
        osp.join(self.root)
        fs.rm(self.raw_dir)

    def build_training_clusters(self, params, debug):
        val_ids = {int(l) for l in open(self.params['VAL']).readlines()}

        if debug:
            val_ids = []

        # read & clean list.csv
        with open(params['LIST']) as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0], r[3], int(r[4])] for r in reader
                    if float(r[2]) <= self.params['RESCUT']
                    and parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        # compile training and validation sets
        train = {}
        valid = {}

        if debug:
            rows = rows[:20]
        for r in rows:
            if r[2] in val_ids:
                if r[2] in valid.keys():
                    valid[r[2]].append(r[:2])
                else:
                    valid[r[2]] = [r[:2]]
            else:
                if r[2] in train.keys():
                    train[r[2]].append(r[:2])
                else:
                    train[r[2]] = [r[:2]]
        if debug:
            valid = train
        return train, valid

    def process_set(self, dataset: str):
        self.train, self.valid = self.build_training_clusters(
            self.params, debug=False)
        if self.set_type == 'train':
            self.item = self.train
        elif self.set_type == 'val':
            self.item = self.valid

        protein_graph = []

        for ID in self.item.keys():

            sel_idx = np.random.randint(0, len(self.item[ID]))

            pdbid, chid = self.item[ID][sel_idx][0].split('_')

            PREFIX = "{}/pdb/{}/{}".format(self.params['DIR'], pdbid[1:3],
                                           pdbid)
            # load metadata

            if not os.path.isfile(PREFIX + ".pt"):
                item_graph = Data(seq=np.zeros(5))
                protein_graph.append(item_graph)
                continue

            meta = torch.load(PREFIX + ".pt")
            asmb_ids = meta['asmb_ids']
            asmb_chains = meta['asmb_chains']
            chids = np.array(meta['chains'])

            asmb_candidates = {
                a
                for a, b in zip(asmb_ids, asmb_chains) if chid in b.split(',')
            }

            if len(asmb_candidates) < 1:
                chain = torch.load(f"{PREFIX}_{chid}.pt")
                L = len(chain['seq'])
                item_graph = Data(
                    seq=chain['seq'],
                    xyz=chain['xyz'],
                    idx=torch.zeros(L).int(),
                    masked=torch.Tensor([0]).int(),
                    #label = self.item[0]
                )

                protein_graph.append(item_graph)

            asmb_i = random.sample(list(asmb_candidates), 1)

            idx = np.where(np.array(asmb_ids) == asmb_i)[0]

            # load relevant chains
            chains = {
                c: torch.load(f"{PREFIX}_{c}.pt")
                for i in idx
                for c in asmb_chains[i] if c in meta['chains']
            }

            asmb = {}
            for k in idx:

                # pick k-th xform
                xform = meta['asmb_xform%d' % k]
                u = xform[:, :3, :3]
                r = xform[:, :3, 3]

                # select chains which k-th xform should be applied to
                s1 = set(meta['chains'])
                s2 = set(asmb_chains[k].split(','))
                chains_k = s1 & s2

                # transform selected chains
                for c in chains_k:
                    try:
                        xyz = chains[c]['xyz']
                        xyz_ru = torch.einsum('bij,raj->brai', u,
                                              xyz) + r[:, None, None, :]
                        asmb.update({
                            (c, k, i): xyz_i
                            for i, xyz_i in enumerate(xyz_ru)
                        })
                    except KeyError:
                        item_graph = Data(seq=np.zeros(5))
                        protein_graph.append(item_graph)
                        continue

            # select chains which share considerable similarity to chid
            seqid = meta['tm'][chids == chid][0, :, 1]
            homo = {
                ch_j
                for seqid_j, ch_j in zip(seqid, chids)
                if seqid_j > params['HOMO']
            }
            # stack all chains in the assembly together
            seq, xyz, idx, masked = "", [], [], []
            seq_list = []
            for counter, (k, v) in enumerate(asmb.items()):
                seq += chains[k[0]]['seq']
                seq_list.append(chains[k[0]]['seq'])
                xyz.append(v)
                idx.append(torch.full((v.shape[0], ), counter))
                if k[0] in homo:
                    masked.append(counter)

            item_graph = Data(
                seq=seq,
                xyz=torch.cat(xyz, dim=0),
                idx=torch.cat(idx, dim=0),
                masked=torch.Tensor(masked).int(),
                #label = self.item[0]
            )

            protein_graph.append(item_graph)

        return protein_graph

    def process(self):
        if self.set_type == 'train':
            self.save(self.process_set('train'), self.processed_paths[0])
        elif self.set_type == 'val':
            self.save(self.process_set('valid'), self.processed_paths[1])
