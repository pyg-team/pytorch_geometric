from itertools import chain

import torch
from torch_sparse import SparseTensor
from scipy.sparse.csgraph import minimum_spanning_tree

from torch_geometric.utils import to_undirected

try:
    import rdkit.Chem as Chem
except ImportError:
    Chem = None


def tree_decomposition(mol):
    r"""The tree decomposition algorithm of molecules from the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://arxiv.org/abs/1802.04364>`_ paper.
    Returns the graph connectivity of the junction tree, the assigmnet mapping
    each atom to its clique in the junction tree, and the number of cliques.

    Args:
        mol (rdkit.Chem.Mol): A :obj:`rdkit` molecule.

    :rtype: (LongTensor, LongTensor, int)
    """

    if Chem is None:
        raise ImportError('Package `rdkit` could not be found.')

    # Cliques = rings and bonds.
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # Generate `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    # Merge rings that share more than 2 atoms as they form bridged compounds.
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    cliques[c2] = []
    cliques = [c for c in cliques if len(c) > 0]

    # Update `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    # Add singleton cliques in case there are more than 2 intersecting
    # cliques. We further compute the "initial" clique graph.
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom]
        if len(cs) <= 1:
            continue

        # Number of bond clusters that the atom lies in.
        bonds = [c for c in cs if len(cliques[c]) == 2]
        # Number of ring clusters that the atom lies in.
        rings = [c for c in cs if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99

        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        row, col = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        clique_graph = SparseTensor(row=row, col=col, value=inv_weight,
                                    sparse_sizes=(len(cliques), len(cliques)))
        junc_tree = minimum_spanning_tree(clique_graph.to_scipy('csr'))
        row, col, _ = SparseTensor.from_scipy(junc_tree).coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique = torch.stack([row, col], dim=0)

    return edge_index, atom2clique, len(cliques)
