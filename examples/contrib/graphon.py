import numpy as np
import torch
from pyg_graphons import Graphon

from torch_geometric.data import Data

"""
    Purpose of this script is to test various properties of the sampled
    graphons given some toy cases. If the sampled graphs obey the
    desired properties (approximately), we can conclude the estimation
    and sampling procedure is correct.
"""
np.random.seed(0)
if __name__ == "__main__":
    ### Test 1 - Sampling from various cliques
    graphon_mod_5 = Graphon(
        graphs=[],
        padding=True,
        r=15,
        align_max_size=100,  # not relevant since we are not doing estimation
        label=0,
        graphon=np.reshape(
            np.array([
                1 if i % 5 == 0 and j % 5 == 0 else 0 for i in range(15)
                for j in range(15)
            ]), (15, 15)))
    graphon_mod_3 = Graphon(
        graphs=[],
        padding=True,
        r=15,
        align_max_size=100,  # not relevant since we are not doing estimation
        label=0,
        graphon=np.reshape(
            np.array([
                1 if i % 3 == 0 and j % 3 == 0 else 0 for i in range(15)
                for j in range(15)
            ]), (15, 15)))
    # both graphons are deterministic - they automatically include an edge if the
    # nodes' values are equal to 0 mod 3 or mod 5, respectively
    # therefore when we sample ~n nodes, depending on which value mod 3 or mod 5 they belong to
    # they should only have an edge between them in those cases
    print(f"Values of graphons")
    print(graphon_mod_5.get_graphon())
    print(graphon_mod_3.get_graphon())
    for _ in range(100):
        # generate graph from each graphon
        edges_3, nodes_3 = graphon_mod_3.generate(50)
        edges_5, nodes_5 = graphon_mod_5.generate(50)
        assert ((nodes_3[[np.nonzero(edges_3)[0],
                          np.nonzero(edges_3)[1]]] % 3 == 0).all())
        assert ((nodes_5[[np.nonzero(edges_5)[0],
                          np.nonzero(edges_5)[1]]] % 5 == 0).all())
        # the above checks that the nodes that have an edge between them both have
        # values that are equal to 0 mod 3 or 5, respectively

    ### Test 2 - Mixup graph statistics
    mixed_graphon = Graphon(
        graphs=[],
        padding=True,
        r=15,
        align_max_size=100,  # not relevant since we are not doing estimation
        label=0,
        graphon=0.5 * graphon_mod_3.get_graphon() +
        0.5 * graphon_mod_5.get_graphon())
    total_frac = 0
    for _ in range(100):
        edges, nodes = mixed_graphon.generate(10)
        # nodes belonging to an edge must either be 0 mod 3 or 0 mod 5
        num_edges = edges.sum()
        possible_edges = (nodes % 5 == 0).sum()**2 + (nodes % 3 == 0).sum()**2
        total_frac += (num_edges / possible_edges) / 100
    print(f"Fraction of possible edges sampled: {total_frac}")
    # total fraction of edges ends up being slightly over 50 percent, which makes sense
    # since now each edge has ~50 percent chance of being sampled except for the edge when
    # eg both nodes have value 0, since this has value 0 mod both 3 and 5

    ### Test 3 - Graphon estimation
    graphs = [
        np.random.binomial(n=1, p=0.5, size=(50, 50)) for _ in range(1000)
    ]
    graphs_data = list(
        map(
            lambda x: Data(edge_index=torch.tensor(np.nonzero(x)), y=torch.
                           LongTensor([0])), graphs))
    estimated_graphon = Graphon(
        graphs=graphs_data,
        padding=True,
        r=10,
        align_max_size=1000,  # not relevant since we are not doing estimation
        label=0,
    )
    print(f"Estimated graphon: {estimated_graphon.get_graphon()}")
    # in this test, we want a graphon that is 0.5 at all non-diagonal values, since we are selecting iid bernoulli edges
    # the estimated graphon has values around 0.6 - this could be a potential issue with the estimation method
    # and is an area for further investigation
