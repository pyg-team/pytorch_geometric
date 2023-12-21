import multiprocessing as mp
import os
from itertools import product
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Voronoi
from sklearn.preprocessing import OneHotEncoder
from torch import from_numpy, long, tensor, where

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_networkx

ADSORBATE_ELEMS = ["C", "H", "O", "N", "S"]


class FGDataset(InMemoryDataset):
    r"""FG-dataset from the `"Fast evaluation of the adsorption energy
    of organic molecules on metals via graph neural networks"
    <https://www.nature.com/articles/s43588-023-00437-y>`_, consisting of 222
    closed-shell organic molecules adsorbed on 14 different transition metals.
    Graphs are generated from the DFT structures stored in the ASE database.

    Args:
        root (str): Root directory where the dataset should be saved.
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
        tol (float, optional): Tolerance (in Angstrom) for the Voronoi analysis
            to detect chemical bonds. (default: :obj:`0.25`)
        sf (float, optional): Scaling factor for the atomic radii
            of the metals for the detection of surface-adsorbate bonds.
            (default: :obj:`1.5`)
        second_order (bool, optional): Whether to include second-order
            surface neighbors in the graph. (default: :obj:`False`)
    """

    url = "https://zenodo.org/records/10410523/files/FGdataset.db?download=1"
    elements = [
        "Ag",
        "Au",
        "Cd",
        "Co",
        "Cu",
        "Fe",
        "Ir",
        "Ni",
        "Os",
        "Pd",
        "Pt",
        "Rh",
        "Ru",
        "Zn",
        "C",
        "H",
        "N",
        "O",
        "S",
    ]
    ohe_elems = OneHotEncoder().fit(np.array(elements).reshape(-1, 1))

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        tol: float = 0.25,
        sf: float = 1.5,
        second_order: bool = False,
    ) -> None:
        self.tol = tol
        self.sf = sf
        self.second_order = second_order
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return "FGdataset.db"

    @property
    def processed_file_names(self) -> str:
        return "FGdataset.pt"

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        from ase.db import connect

        db = connect(self.raw_paths[0])
        args_list = []
        for row in db.select():
            args_list.append(
                (row, self.tol, self.sf, self.second_order, self.ohe_elems))

        with mp.Pool(os.cpu_count() // 2) as pool:
            data_list = pool.starmap(row_to_data, args_list)
        data_list = [data for data in data_list if fg_filter(data)]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


def row_to_data(row, tol: float, scaling_factor: float, second_order: bool,
                ohe_elems: OneHotEncoder) -> Data:
    """Create Data object from ASE database row.

    Args:
        row (AtomsRow): ASE database row.
        tol (float): tolerance for distance between two atoms to be connected.
        scaling_factor (float): scaling factor for the surface atoms' radii.
        second_order (bool): Include second-order surface neighbours.
        ohe_elems (OneHotEncoder): one-hot encoder for the chemical elements.

    Returns:
        graph (Data): PyG Data object.
    """
    idxs, elems, nl = atoms_to_graph(row.toatoms(), tol, scaling_factor,
                                     second_order)
    elem_array = np.array(list(elems)).reshape(-1, 1)
    elem_enc = ohe_elems.transform(elem_array).toarray()
    x = from_numpy(elem_enc).float()
    edges = [(idxs.index(pair[0]), idxs.index(pair[1])) for pair in nl]
    edge_tails = [x for x, _ in edges] + [y for _, y in edges]
    edge_heads = [y for _, y in edges] + [x for x, _ in edges]
    edge_index = tensor([edge_tails, edge_heads], dtype=long)
    graph = Data(
        x=x,
        edge_index=edge_index,
        ase_atoms=row.toatoms(),
        formula=row.get("formula"),
        metal=row.get("metal"),
        facet=row.get("facet"),
        energy=row.get("energy"),
        scaled_energy=row.get("scaled_energy"),
        e_ads=row.get("e_ads"),
        e_mol=row.get("e_mol"),
        e_slab=row.get("e_slab"),
        node_feats=list(ohe_elems.categories_[0]),
        note=row.get("note"),
    )
    print("Graph generated for {}\n".format(graph.formula))
    return graph


def atoms_to_graph(atoms, tol: float, scaling_factor: float,
                   second_order: bool) -> tuple:
    """Get nodes and edges from ASE Atoms object.

    Args:
        atoms (Atoms): ASE Atoms object of the adsorbate-surface system.
        tol (float): tolerance for the distance between
                     two atoms to be considered connected.
        scaling_factor (float): scaling factor for the surface atoms' radii.
        second_order (bool): Include second-order surface neighbours.

    Returns:
        tuple[list[int], list[str], list[tuple[int, int]]]:
            - list[int]: indices of the atoms in the ensemble.
            - list[str]: chemical elements of the atoms in the ensemble.
            - list[tuple[int, int]]: connectivity list of the ensemble.
    """
    adsorbate_idxs = {
        atom.index
        for atom in atoms if atom.symbol in ADSORBATE_ELEMS
    }
    nl = get_voronoi_neighborlist(atoms, tol, scaling_factor)
    surface_idxs = {
        pair[1] if pair[0] in adsorbate_idxs else pair[0]
        for pair in nl
        if (pair[0] in adsorbate_idxs and pair[1] not in adsorbate_idxs) or (
            pair[1] in adsorbate_idxs and pair[0] not in adsorbate_idxs)
    }

    if second_order:
        surface_idxs = surface_idxs.union({
            pair[1] if pair[0] in surface_idxs else pair[0]
            for pair in nl
            if (pair[0] in surface_idxs and pair[1] not in adsorbate_idxs) or (
                pair[1] in surface_idxs and pair[0] not in adsorbate_idxs)
        })

    idxs = list(adsorbate_idxs.union(surface_idxs))
    elems = [atoms[index].symbol for index in idxs]
    neighborlist = [pair for pair in nl if pair[0] in idxs and pair[1] in idxs]
    return idxs, elems, neighborlist


def get_voronoi_neighborlist(atoms, tol: float, sf: float) -> np.ndarray:
    """Get connectivity list considering periodic boundary conditions.

    Args:
        atoms (Atoms): Atoms object representing the adsorbate-metal surface.
        tol (float): Tolerance for the distance criterion
                     between two atoms to be considered connected.
        sf (float): Scaling factor for the covalent radii of the metal atoms.

    Returns:
        np.ndarray: connectivity of the system.
                    Each row represents a pair of connected atoms.
    """
    RADII = {
        "Cd": 1.44,
        "C": 0.76,
        "Co": 1.50,
        "Cu": 1.32,
        "Au": 1.36,
        "H": 0.31,
        "Ir": 1.41,
        "Fe": 1.52,
        "Ni": 1.24,
        "N": 0.71,
        "Os": 1.44,
        "O": 0.66,
        "Pd": 1.39,
        "Pt": 1.36,
        "Rh": 1.42,
        "Ru": 1.46,
        "Ag": 1.45,
        "S": 1.05,
        "Zn": 1.22,
    }  # Angstrom, from Cordero et al.

    coords_arr = np.repeat(
        np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27,
        axis=0)
    mirrors = np.repeat(
        np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1),
        coords_arr.shape[1],
        axis=1,
    )
    corrected_coords = np.reshape(
        coords_arr + mirrors,
        (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]),
    )
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr,
                           np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]),
                           axis=0)
    pairs = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = (RADII[atoms[pair[0]].symbol] +
                     RADII[atoms[pair[1]].symbol] + tol + (sf - 1.0) *
                     ((atoms[pair[0]].symbol not in ADSORBATE_ELEMS) *
                      RADII[atoms[pair[0]].symbol] +
                      (atoms[pair[1]].symbol not in ADSORBATE_ELEMS) *
                      RADII[atoms[pair[1]].symbol]))
        if distance <= threshold:
            pairs.append(pair)
    return np.sort(np.array(pairs), axis=1)


def fragment_filter(graph: Data) -> bool:
    """Check adsorbate fragmentation in the graph.

    Args:
        graph(Data): Adsorption graph.

    Returns:
        (bool): True = Adsorbate not fragmented in the graph
                False = Adsorbate fragmented in the graph
    """
    from networkx import is_connected

    adsorbate_elems_idxs = [
        graph.node_feats.index(elem) for elem in ADSORBATE_ELEMS
    ]
    adsorbate_nodes = []
    for node_idx in range(graph.num_nodes):
        idx = where(graph.x[node_idx, :] == 1)[0].item()
        if idx in adsorbate_elems_idxs:
            adsorbate_nodes.append(node_idx)
    subgraph = graph.subgraph(tensor(adsorbate_nodes))
    graph_nx = to_networkx(subgraph, to_undirected=True)
    if subgraph.num_nodes != 1 and subgraph.num_edges != 0:
        if is_connected(graph_nx):
            return True
        else:
            print(f"{graph.formula}: Fragmented adsorbate.\n".format(
                graph.formula))
            return False
    else:
        return True


def H_filter(graph: Data) -> bool:
    """Check H connectivity.
    H atoms must be connected to maximum one adsorbate atom.

    Args:
        graph(Data): Adsorption graph.

    Returns:
        (bool): True = Correct connectivity for all H atoms in the adsorbate
                False = Bad connectivity for at least one H atom
    """
    adsorbate_elems_idxs = [
        graph.node_feats.index(elem) for elem in ADSORBATE_ELEMS
    ]
    H_idx = graph.node_feats.index("H")
    H_nodes_idxs = []
    for i in range(graph.num_nodes):
        if graph.x[i, H_idx] == 1:
            H_nodes_idxs.append(i)
    for node_idx in H_nodes_idxs:
        counter = 0  # edges between H and adsorbate atoms
        for j in range(graph.num_edges):
            if node_idx == graph.edge_index[0, j]:
                other_atom = where(
                    graph.x[graph.edge_index[1, j], :] == 1)[0].item()
                counter += 1 if other_atom in adsorbate_elems_idxs else 0
        if counter > 1:
            print(f"{graph.formula}: Wrong H connectivity.\n")
            return False
    return True


def C_filter(graph: Data) -> bool:
    """Check C connectivity.
    C atoms must be connected to maximum 4 adsorbate atoms.

    Args:
        graph(Data): Adsorption graph.

    Returns:
        (bool): True = Correct connectivity for all C atoms in the adsorbate
                False = Bad connectivity for at least one C atom
    """
    adsorbate_elems_idxs = [
        graph.node_feats.index(elem) for elem in ADSORBATE_ELEMS
    ]
    C_idx = adsorbate_elems_idxs[graph.node_feats.index("C")]
    C_nodes_idxs = [
        i for i in range(graph.num_nodes) if graph.x[i, C_idx] == 1
    ]
    for node_idx in C_nodes_idxs:
        counter = 0  # edges between C and adsorbate atoms
        for j in range(graph.num_edges):
            if node_idx == graph.edge_index[0, j]:
                other_atom = where(
                    graph.x[graph.edge_index[1, j], :] == 1)[0].item()
                counter += 1 if other_atom in adsorbate_elems_idxs else 0
        if counter > 4:
            print(f"{graph.formula}: Wrong C connectivity.\n")
            return False
    return True


def adsorption_filter(graph: Data) -> bool:
    """Check surface presence in the adsorption graph.

    Args:
        graph(Data): Adsorption graph.

    Returns:
        (bool): True = Metal presence in the graph
                False = Metal absence in the graph
    """
    if graph.metal == "N/A":  # gas-phase molecule
        return True
    adsorbate_elems_idxs = [
        graph.node_feats.index(elem) for elem in ADSORBATE_ELEMS
    ]
    for node_idx in range(graph.num_nodes):
        idx = where(graph.x[node_idx, :] == 1)[0].item()
        if idx not in adsorbate_elems_idxs:
            return True
    print(f"{graph.formula}: No surface representation.\n")
    return False


def fg_filter(graph: Data) -> bool:
    return (fragment_filter(graph) and H_filter(graph) and C_filter(graph)
            and adsorption_filter(graph))
