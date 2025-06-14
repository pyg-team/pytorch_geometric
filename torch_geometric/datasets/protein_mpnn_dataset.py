import os
import pickle
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class ProteinMPNNDataset(InMemoryDataset):
    r"""The ProteinMPNN dataset from the `"Robust deep learning based protein
    sequence design using ProteinMPNN"
    <https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        size (str): Size of the PDB information to train the model.
            If :obj:`"small"`, loads the small dataset (229.4 MB).
            If :obj:`"large"`, loads the large dataset (64.1 GB).
            (default: :obj:`"small"`)
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"valid"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        datacut (str, optional): Date cutoff to filter the dataset.
            (default: :obj:`"2030-01-01"`)
        rescut (float, optional): PDB resolution cutoff.
            (default: :obj:`3.5`)
        homo (float, optional): Homology cutoff.
            (default: :obj:`0.70`)
        max_length (int, optional): Maximum length of the protein complex.
            (default: :obj:`10000`)
        num_units (int, optional): Number of units of the protein complex.
            (default: :obj:`150`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    raw_url = {
        'small':
        'https://files.ipd.uw.edu/pub/training_sets/'
        'pdb_2021aug02_sample.tar.gz',
        'large':
        'https://files.ipd.uw.edu/pub/training_sets/'
        'pdb_2021aug02.tar.gz',
    }

    splits = {
        'train': 1,
        'valid': 2,
        'test': 3,
    }

    def __init__(
        self,
        root: str,
        size: str = 'small',
        split: str = 'train',
        datacut: str = '2030-01-01',
        rescut: float = 3.5,
        homo: float = 0.70,
        max_length: int = 10000,
        num_units: int = 150,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.size = size
        self.split = split
        self.datacut = datacut
        self.rescut = rescut
        self.homo = homo
        self.max_length = max_length
        self.num_units = num_units

        self.sub_folder = self.raw_url[self.size].split('/')[-1].split('.')[0]

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[self.splits[self.split]])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.sub_folder}/{f}'
            for f in ['list.csv', 'valid_clusters.txt', 'test_clusters.txt']
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['splits.pkl', 'train.pt', 'valid.pt', 'test.pt']

    def download(self) -> None:
        file_path = download_url(self.raw_url[self.size], self.raw_dir)
        extract_tar(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self) -> None:
        alphabet_set = set(list('ACDEFGHIKLMNPQRSTVWYX'))
        cluster_ids = self._process_split()
        total_items = sum(len(items) for items in cluster_ids.values())
        data_list = []

        with tqdm(total=total_items, desc="Processing") as pbar:
            for _, items in cluster_ids.items():
                for chain_id, _ in items:
                    item = self._process_pdb1(chain_id)

                    if 'label' not in item:
                        pbar.update(1)
                        continue
                    if len(list(np.unique(item['idx']))) >= 352:
                        pbar.update(1)
                        continue

                    my_dict = self._process_pdb2(item)

                    if len(my_dict['seq']) > self.max_length:
                        pbar.update(1)
                        continue
                    bad_chars = set(list(
                        my_dict['seq'])).difference(alphabet_set)
                    if len(bad_chars) > 0:
                        pbar.update(1)
                        continue

                    x_chain_all, chain_seq_label_all, mask, chain_mask_all, residue_idx, chain_encoding_all = self._process_pdb3(  # noqa: E501
                        my_dict)

                    data = Data(
                        x=x_chain_all,  # [seq_len, 4, 3]
                        chain_seq_label=chain_seq_label_all,  # [seq_len]
                        mask=mask,  # [seq_len]
                        chain_mask_all=chain_mask_all,  # [seq_len]
                        residue_idx=residue_idx,  # [seq_len]
                        chain_encoding_all=chain_encoding_all,  # [seq_len]
                    )

                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                    if len(data_list) >= self.num_units:
                        pbar.update(total_items - pbar.n)
                        break
                    pbar.update(1)
                else:
                    continue
                break
            self.save(data_list, self.processed_paths[self.splits[self.split]])

    def _process_split(self) -> Dict[int, List[Tuple[str, int]]]:
        import pandas as pd
        save_path = self.processed_paths[0]

        if os.path.exists(save_path):
            print('Load split')
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            # CHAINID, DEPOSITION, RESOLUTION, HASH, CLUSTER, SEQUENCE
            df = pd.read_csv(self.raw_paths[0])
            df = df[(df['RESOLUTION'] <= self.rescut)
                    & (df['DEPOSITION'] <= self.datacut)]

            val_ids = pd.read_csv(self.raw_paths[1], header=None)[0].tolist()
            test_ids = pd.read_csv(self.raw_paths[2], header=None)[0].tolist()

            # compile training and validation sets
            data = {
                'train': defaultdict(list),
                'valid': defaultdict(list),
                'test': defaultdict(list),
            }

            for _, r in tqdm(df.iterrows(), desc='Processing split',
                             total=len(df)):
                cluster_id = r['CLUSTER']
                hash_id = r['HASH']
                chain_id = r['CHAINID']
                if cluster_id in val_ids:
                    data['valid'][cluster_id].append((chain_id, hash_id))
                elif cluster_id in test_ids:
                    data['test'][cluster_id].append((chain_id, hash_id))
                else:
                    data['train'][cluster_id].append((chain_id, hash_id))

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

        return data[self.split]

    def _process_pdb1(self, chain_id: str) -> Dict[str, Any]:
        pdbid, chid = chain_id.split('_')
        prefix = f'{self.raw_dir}/{self.sub_folder}/pdb/{pdbid[1:3]}/{pdbid}'
        # load metadata
        if not os.path.isfile(f'{prefix}.pt'):
            return {'seq': np.zeros(5)}
        meta = torch.load(f'{prefix}.pt')
        asmb_ids = meta['asmb_ids']
        asmb_chains = meta['asmb_chains']
        chids = np.array(meta['chains'])

        # find candidate assemblies which contain chid chain
        asmb_candidates = {
            a
            for a, b in zip(asmb_ids, asmb_chains) if chid in b.split(',')
        }

        # if the chains is missing is missing from all the assemblies
        # then return this chain alone
        if len(asmb_candidates) < 1:
            chain = torch.load(f'{prefix}_{chid}.pt')
            L = len(chain['seq'])
            return {
                'seq': chain['seq'],
                'xyz': chain['xyz'],
                'idx': torch.zeros(L).int(),
                'masked': torch.Tensor([0]).int(),
                'label': chain_id,
            }

        # randomly pick one assembly from candidates
        asmb_i = random.sample(list(asmb_candidates), 1)

        # indices of selected transforms
        idx = np.where(np.array(asmb_ids) == asmb_i)[0]

        # load relevant chains
        chains = {
            c: torch.load(f'{prefix}_{c}.pt')
            for i in idx
            for c in asmb_chains[i] if c in meta['chains']
        }

        # generate assembly
        asmb = {}
        for k in idx:

            # pick k-th xform
            xform = meta[f'asmb_xform{k}']
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
                    xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:, None,
                                                                       None, :]
                    asmb.update({
                        (c, k, i): xyz_i
                        for i, xyz_i in enumerate(xyz_ru)
                    })
                except KeyError:
                    return {'seq': np.zeros(5)}

        # select chains which share considerable similarity to chid
        seqid = meta['tm'][chids == chid][0, :, 1]
        homo = {
            ch_j
            for seqid_j, ch_j in zip(seqid, chids) if seqid_j > self.homo
        }
        # stack all chains in the assembly together
        seq: str = ''
        xyz_all: List[torch.Tensor] = []
        idx_all: List[torch.Tensor] = []
        masked: List[int] = []
        seq_list = []
        for counter, (k, v) in enumerate(asmb.items()):
            seq += chains[k[0]]['seq']
            seq_list.append(chains[k[0]]['seq'])
            xyz_all.append(v)
            idx_all.append(torch.full((v.shape[0], ), counter))
            if k[0] in homo:
                masked.append(counter)

        return {
            'seq': seq,
            'xyz': torch.cat(xyz_all, dim=0),
            'idx': torch.cat(idx_all, dim=0),
            'masked': torch.Tensor(masked).int(),
            'label': chain_id,
        }

    def _process_pdb2(self, t: Dict[str, Any]) -> Dict[str, Any]:
        init_alphabet = list(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet + extra_alphabet
        my_dict: Dict[str, Union[str, int, Dict[str, Any], List[Any]]] = {}
        concat_seq = ''
        mask_list = []
        visible_list = []
        for idx in list(np.unique(t['idx'])):
            letter = chain_alphabet[idx]
            res = np.argwhere(t['idx'] == idx)
            initial_sequence = "".join(list(
                np.array(list(t['seq']))[res][
                    0,
                ]))
            if initial_sequence[-6:] == "HHHHHH":
                res = res[:, :-6]
            if initial_sequence[0:6] == "HHHHHH":
                res = res[:, 6:]
            if initial_sequence[-7:-1] == "HHHHHH":
                res = res[:, :-7]
            if initial_sequence[-8:-2] == "HHHHHH":
                res = res[:, :-8]
            if initial_sequence[-9:-3] == "HHHHHH":
                res = res[:, :-9]
            if initial_sequence[-10:-4] == "HHHHHH":
                res = res[:, :-10]
            if initial_sequence[1:7] == "HHHHHH":
                res = res[:, 7:]
            if initial_sequence[2:8] == "HHHHHH":
                res = res[:, 8:]
            if initial_sequence[3:9] == "HHHHHH":
                res = res[:, 9:]
            if initial_sequence[4:10] == "HHHHHH":
                res = res[:, 10:]
            if res.shape[1] >= 4:
                chain_seq = "".join(list(np.array(list(t['seq']))[res][0]))
                my_dict[f'seq_chain_{letter}'] = chain_seq
                concat_seq += chain_seq
                if idx in t['masked']:
                    mask_list.append(letter)
                else:
                    visible_list.append(letter)
                coords_dict_chain = {}
                all_atoms = np.array(t['xyz'][res])[0]  # [L, 14, 3]
                for i, c in enumerate(['N', 'CA', 'C', 'O']):
                    coords_dict_chain[
                        f'{c}_chain_{letter}'] = all_atoms[:, i, :].tolist()
                my_dict[f'coords_chain_{letter}'] = coords_dict_chain
        my_dict['name'] = t['label']
        my_dict['masked_list'] = mask_list
        my_dict['visible_list'] = visible_list
        my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
        my_dict['seq'] = concat_seq
        return my_dict

    def _process_pdb3(
        self, b: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        L = len(b['seq'])
        # residue idx with jumps across chains
        residue_idx = -100 * np.ones([L], dtype=np.int32)
        # get the list of masked / visible chains
        masked_chains, visible_chains = b['masked_list'], b['visible_list']
        visible_temp_dict, masked_temp_dict = {}, {}
        for letter in masked_chains + visible_chains:
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        # check for duplicate chains (same sequence but different identity)
        for _, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        # build protein data structures
        all_chains = masked_chains + visible_chains
        np.random.shuffle(all_chains)
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c, l0, l1 = 1, 0, 0
        for letter in all_chains:
            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = b[f'coords_chain_{letter}']
            x_chain = np.stack([
                chain_coords[c] for c in [
                    f'N_chain_{letter}', f'CA_chain_{letter}',
                    f'C_chain_{letter}', f'O_chain_{letter}'
                ]
            ], 1)  # [chain_length, 4, 3]
            x_chain_list.append(x_chain)
            chain_seq_list.append(chain_seq)
            if letter in visible_chains:
                chain_mask = np.zeros(chain_length)  # 0 for visible chains
            elif letter in masked_chains:
                chain_mask = np.ones(chain_length)  # 1 for masked chains
            chain_mask_list.append(chain_mask)
            chain_encoding_list.append(c * np.ones(chain_length))
            l1 += chain_length
            residue_idx[l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1
        x_chain_all = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        chain_seq_all = "".join(chain_seq_list)
        # [L,] 1.0 for places that need to be predicted
        chain_mask_all = np.concatenate(chain_mask_list, 0)
        chain_encoding_all = np.concatenate(chain_encoding_list, 0)

        # Convert to labels
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        chain_seq_label_all = np.asarray(
            [alphabet.index(a) for a in chain_seq_all], dtype=np.int32)

        isnan = np.isnan(x_chain_all)
        mask = np.isfinite(np.sum(x_chain_all, (1, 2))).astype(np.float32)
        x_chain_all[isnan] = 0.

        # Conversion
        return (
            torch.from_numpy(x_chain_all).to(dtype=torch.float32),
            torch.from_numpy(chain_seq_label_all).to(dtype=torch.long),
            torch.from_numpy(mask).to(dtype=torch.float32),
            torch.from_numpy(chain_mask_all).to(dtype=torch.float32),
            torch.from_numpy(residue_idx).to(dtype=torch.long),
            torch.from_numpy(chain_encoding_all).to(dtype=torch.long),
        )
