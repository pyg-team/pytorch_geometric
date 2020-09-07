import pandas as pd
import numpy as np
import torch
import logging
from torch_geometric.data import Data


def process_event(event, pt_min=0, phi_range=(-np.pi, np.pi),
                  eta_range=(-5, 5), n_phi_sections=9, n_eta_sections=3,
                  phi_slope_max=6e-4, z0_max=100):
    # parameters from heptrkx-gnn-tracking/configs/prep_big.yaml
    event_id, hits, cells, particles, truth = event
    # Barrel volume and layer ids
    vlids = [(8, 2), (8, 4), (8, 6), (8, 8),
             (13, 2), (13, 4), (13, 6), (13, 8),
             (17, 2), (17, 4)]
    n_det_layers = len(vlids)

    hits = select_hits(hits, truth, particles, vlids,
                       pt_min).assign(event_id=event_id)

    # Divide detector into sections
    phi_edges = np.linspace(*phi_range, num=n_phi_sections)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections)

    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])

    # Define adjacent layers
    layer_pairs = np.stack([np.arange(n_det_layers)[:-1],
                            np.arange(n_det_layers)[1:]], axis=1)

    graphs = [construct_graph(section_hits, layer_pairs, phi_slope_max,
                              z0_max, feature_names, feature_scale)
              for section_hits in hits_sections]
    return graphs


def select_hits(hits, truth, particles, vlids, pt_min=0):
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])

    n_det_layers = len(vlids)

    ''' Losing information in hits '''
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])

    # Calculate particle transverse momentum
    pt = np.sqrt(particles.px**2 + particles.py**2)
    # True particle selection.
    # Applies pt cut, removes all noise hits.
    particles = particles[pt > pt_min]
    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id']], on='particle_id'))
    # Calculate derived hits variables
    ''' vector length r, angle phi in radians between x, y'''
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)

    ''' r, phi instead of x, y and merging particle_id '''
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
    # Remove duplicate hits
    hits = hits.loc[
        hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    ]
    return hits


def calc_eta(r, z):
    ''' angle theta in radians between r, z '''
    theta = np.arctan2(r, z)
    # ??
    return -1. * np.log(np.tan(theta / 2.))


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi, pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi
    return dphi


def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i + 1]
        # Select hits in this phi section
        ''' hits with similar angle between x,y '''
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]

        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j + 1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))
    return hits_sections


def select_segments(hits1, hits2, phi_slope_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.
    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['event_id', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='event_id', suffixes=('_1', '_2'))

    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr

    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

    # Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    return hit_pairs[['index_1', 'index_2']][good_seg_mask]


def construct_graph(hits, layer_pairs, phi_slope_max, z0_max,
                    feature_names, feature_scale):
    """Construct one graph (e.g. from one event)"""

    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        segments.append(select_segments(hits1, hits2, phi_slope_max, z0_max))
    # Combine segments from all layer pairs
    segments = pd.concat(segments)

    # Prepare the graph matrices
    n_hits = len(hits)
    n_edges = len(segments)

    x = (hits[feature_names].values / feature_scale).astype(np.float32)
    ''' not necessary for edge_index
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    '''
    y = np.zeros(n_edges, dtype=np.float32)

    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)

    ''' positional index for remaining hit pairs '''
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values

    '''
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    '''

    # Fill the segment labels
    pid1 = hits.particle_id.loc[segments.index_1].values
    pid2 = hits.particle_id.loc[segments.index_2].values
    y[:] = (pid1 == pid2)
    # Return a tuple of the results

    edge_index = torch.from_numpy(np.array([seg_start, seg_end]))
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return Data(x=x, num_nodes=n_hits, edge_index=edge_index, y=y)
