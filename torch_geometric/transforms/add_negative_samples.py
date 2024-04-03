import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

SEED = 20240403


class SampleNegatives(BaseTransform):
    r"""Handles negative edge sampling for inductive graph learning.

    Args:
        edges = all positive edge pairs as a list of tuples [(u1, v1),...]
        datasplit = how the data was partitioned.
            - type: Literal(['source', 'target', 'pair'])
            - if 'source': source nodes are split into train/valid/test, and
            all edges containing those nodes are subsequently partitioned into
            these groups
            - if 'target': same as above, but split by target node ids.
            - if 'pair': no longer inductive learning since no nodes are
            intentionally left out of training. The option is included such
            that the transform is generalizable.
        ratio = the ratio of negative sampled edges to positive edges

    """
    def __init__(self, edges, datasplit, ratio=1):
        self.edges = edges
        self.datasplit = datasplit
        self.ratio = ratio

    def forward(self, data: HeteroData):
        num_pos = len(data['binds'].edge_label)
        num_pos *= self.ratio

        if self.datasplit == 'source':
            subgraph_src = data['binds'].edge_label_index[0].unique()
            global_src = data['source'].node_id[subgraph_src]

            subgraph_tgt = torch.arange(len(data['target'].node_id))
            global_tgt = data['target'].node_id

        elif self.datasplit == 'target':
            subgraph_src = torch.arange(len(data['source'].node_id))
            global_src = data['source'].node_id

            subgraph_tgt = data['binds'].edge_label_index[1].unique()
            global_tgt = data['target'].node_id[subgraph_tgt]

        elif self.datasplit == 'pair':
            subgraph_src = torch.arange(len(data['source'].node_id))
            global_src = data['source'].node_id

            subgraph_tgt = torch.arange(len(data['target'].node_id))
            global_tgt = data['target'].node_id

        subgraph_src = subgraph_src.cpu().numpy()
        global_src = global_src.cpu().numpy()
        subgraph_tgt = subgraph_tgt.cpu().numpy()
        global_tgt = global_tgt.cpu().numpy()

        pos_edges = pd.MultiIndex.from_arrays(self.edges)

        # 3 chances to sample negative edges
        rng = np.random.default_rng(SEED)
        for _ in range(3):
            rnd_srcs = rng.choice(global_src, size=(num_pos * 2))
            rnd_tgts = rng.choice(global_tgt, size=(num_pos * 2))

            rnd_pairs = np.stack((rnd_srcs, rnd_tgts))
            rnd_pairs = np.unique(rnd_pairs, axis=1)
            rnd_pairs = pd.MultiIndex.from_arrays(rnd_pairs)
            inter = rnd_pairs.intersection(pos_edges, sort=False)
            neg_pairs = rnd_pairs.difference(inter, sort=False)

            if len([*neg_pairs]) < num_pos:
                continue
            neg_pairs = rng.choice([*neg_pairs], num_pos, replace=False).T
            break

        else:
            raise RuntimeError("Could not successfully sample negatives.")

        # build dictionaries to map global edge indices
        # to local (subgraph) indices
        source_map = dict(zip(pd.Series(global_src), pd.Series(subgraph_src)))
        target_map = dict(zip(pd.Series(global_tgt), pd.Series(subgraph_tgt)))

        neg_edges_srcs = pd.Series(neg_pairs[0]).map(source_map).values
        neg_edges_tgts = pd.Series(neg_pairs[1]).map(target_map).values

        new_labels = torch.cat(
            (data['binds'].edge_label, torch.Tensor(np.zeros(num_pos))))
        new_edges = torch.cat(
            (data['binds'].edge_label_index,
             torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts]))),
            axis=1).type(torch.int32)

        data['binds'].edge_label = new_labels
        data['binds'].edge_label_index = new_edges

        return data
