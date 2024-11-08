import numpy as np
import logging
import torch_geometric
from torch_geometric import Data, Dataset, InMemoryDataset

from src.pyg_graphons import Graphon
from collections import defaultdict


class MixupDataset(InMemoryDataset):
    
    def __init__(self, dataset : Dataset, specs : list[(int, int, float, int)] = None, estimation_method : str = 'usvt', **args):
        """
        dataset (torch_geometric.Dataset)
        specs (list[(int, int, float, int)]) - list of tuples(label_class_i, label_class_j, mixup_fraction, num_samples_to_generate)
        estimation_method (string): `usvt', `sas', `sba', `mc', `lg'
        args (optional): additional args used in estimation method

        """

        self.dataset = dataset
        self.specs = specs 
        self.estimation_method = estimation_method
        
        if self.specs:
            self.process()
        
        # Reuse already created graphons
        self._graphons = defaultdict(Graphon)
        self._K = 15 # TODO: update this to be dynamic

        # Represent the data for __getitem__
        self.graphs = dataset



    def _mixup(self, class_i : Graphon, class_j : Graphon, mixup_fraction : int) -> Graphon: 
        mixed_graphon = class_i * mixup_fraction + class_j + (1 - mixup_fraction)
        return mixed_graphon
         


    def process(self) -> None:
        n = len(self.specs) 

        for class_i, class_j, mixup_fraction, num_samples in self.specs: 
            logging.info(f"Mixing up graphons for {class_i}, {class_j}")

            # Generate graphon for class_i 
            if class_i in self._graphons:
                class_i_graphon = self._graphons[class_i]
            else:
                # TODO: get all the graphs from self.data that are of class_i
                # Pass this in as a parameter for the graphon constructor 

                class_i_graphon = Graphon()
                self._graphons[class_i] = class_i_graphon


            # Generate graphon for class_j 
            if class_j in self._graphons:
                class_j_graphon = self._graphons[class_j]
            else:
                # TODO: get all the graphs from self.data that are of class_i
                # Pass this in as a parameter for the graphon constructor 

                class_j_graphon = Graphon()
                self._graphons[class_j] = class_j_graphon


            # Call mixup function with mixup_fraction
            mixup_graphon = self._mixup(class_i_graphon, class_j_graphon, mixup_fraction)


            # Generate num_samples samples and add to self.dataset
            for _ in range(num_samples):
                new_sample = mixup_graphon.generate(self._K)
                self.graphs.append(new_sample)


    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        

        if idx < len(self.dataset):
            return self.dataset[idx]
        
        else:
            return self.new_graphs[idx - len(self.dataset)]

# Initial skeleton of estimation taken from https://github.com/eomeragic1/g-mixup-reproducibility/tree/main
# TODO: Fix and customize some implementation details and clean up, annotate, document for PR
# In the paper they said that they used LG as the step function approximator, check with authors !!
def universal_svd(aligned_graphs: List[Tensor], threshold: float = 2.02, sum_graph: Tensor = None) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.
    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """

    if sum_graph is None:
        aligned_graphs = graph_numpy2tensor(aligned_graphs)
        num_graphs = aligned_graphs.size(0)

        if num_graphs > 1:
            sum_graph = torch.mean(aligned_graphs, dim=0)
        else:
            sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)
    print('Doing SVD of matrix of size: ', num_nodes)
    u, s, v = torch.svd(sum_graph)
    print('Finished SVD!')
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def largest_gap(aligned_graphs: List[Tensor], k: int, sum_graph: Tensor = None) -> np.ndarray:
    """
    Estimate a graphon by a stochastic block model based n empirical degrees
    Reference:
    Channarond, Antoine, Jean-Jacques Daudin, and Stéphane Robin.
    "Classification and estimation in the Stochastic Blockmodel based on the empirical degrees."
    Electronic Journal of Statistics 6 (2012): 2574-2601.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param k: the number of blocks
    :return: a (r, r) estimation of graphon
    """
    if sum_graph is None:
        aligned_graphs = graph_numpy2tensor(aligned_graphs)
        num_graphs = aligned_graphs.size(0)

        if num_graphs > 1:
            sum_graph = torch.mean(aligned_graphs, dim=0)
        else:
            sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    # sort node degrees
    degree = torch.sum(sum_graph, dim=1)
    sorted_degree = degree / (num_nodes - 1)
    idx = torch.arange(0, num_nodes)

    # find num_blocks-1 largest gap of the node degrees
    diff_degree = sorted_degree[1:] - sorted_degree[:-1]
    _, index = torch.topk(diff_degree, k=k - 1)
    sorted_index, _ = torch.sort(index + 1, descending=False)
    blocks = {}
    for b in range(k):
        if b == 0:
            blocks[b] = idx[0:sorted_index[b]]
        elif b == k - 1:
            blocks[b] = idx[sorted_index[b - 1]:num_nodes]
        else:
            blocks[b] = idx[sorted_index[b - 1]:sorted_index[b]]

    # derive the graphon by stochastic block model
    probability = torch.zeros(k, k)
    graphon = torch.zeros(num_nodes, num_nodes)
    for i in range(k):
        for j in range(k):
            rows = blocks[i]
            cols = blocks[j]
            tmp = sum_graph[rows, :]
            tmp = tmp[:, cols]
            probability[i, j] = torch.sum(tmp) / (rows.size(0) * cols.size(0))
            for r in range(rows.size(0)):
                for c in range(cols.size(0)):
                    graphon[rows[r], cols[c]] = probability[i, j]
    graphon = graphon.numpy()
    return graphon

def align_graphs(graphs: List[Data],
                 padding: bool = False, N: int = None) -> Tuple[List[Tensor], List[Tensor], int, int, Tensor]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees
    What this function does is it orders each graph adjacency matrix so that the degrees are sorted starting from
    highest to lowest

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :param N: whether to cut the graphs at size N (keeping highest-degree nodes)
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].num_nodes for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    if N:
        sum_graph = np.zeros((N, N))
    else:
        sum_graph = np.zeros((max_num, max_num))
    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].num_nodes
        adj = to_dense_adj(graphs[i].edge_index)[0].numpy()

        node_degree = 0.5 * np.sum(adj, axis=0) + 0.5 * np.sum(adj, axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(adj)
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            if N:
                normalized_node_degrees.append(normalized_node_degree[:N, :])
                sum_graph += aligned_graph[:N, :N]
                aligned_graphs.append(dense_to_sparse(torch.from_numpy(aligned_graph[:N, :N]))[0])
            else:
                normalized_node_degrees.append(normalized_node_degree)
                sum_graph += aligned_graph
                aligned_graphs.append(dense_to_sparse(torch.from_numpy(aligned_graph))[0])
        else:
            if N:
                normalized_node_degrees.append(sorted_node_degree[:N, :])
                sum_graph += sorted_graph[:N, :N]
                aligned_graphs.append(dense_to_sparse(torch.from_numpy(sorted_graph[:N, :N]))[0])
            else:
                normalized_node_degrees.append(sorted_node_degree)
                sum_graph += sorted_graph
                aligned_graphs.append(dense_to_sparse(torch.from_numpy(sorted_graph))[0])

    return aligned_graphs, normalized_node_degrees, max_num, min_num, torch.from_numpy(sum_graph/len(graphs))


class Graphon:
    r"""The Graphon objects as defined in `G-Mixup 
    <https://arxiv.org/pdf/2202.07179>`_ paper.

    Graphons are defined as functions W : [0, 1]² → [0, 1] representing the 
    probability of an edge between two labeled vertices: W(i, j) = P [(i, j) ∈ E]. 
    
    The Graphon class should thus store this distribution in some way. It does so
    using a matrix of probabilities: an r x r matrix, where element (i, j) 
    represents the Bernoulli probability that the edge (i, j) occurs in a 
    representative graph with r nodes. On initialization, the class computes and 
    stores this distribution.

    Args:
        graphs (list[np.ndarray]): Adjacency matrices of the graphs to estimate.
        padding (bool): used while aligning the graphs to the same size
        r (int): representing dimensionality of output (r x r)
        align_max_size (int): hyperparameter # TODO
        label (int): class of the data
        estimation_method (string): `usvt' (default), `sas', `sba', `mc', `lg'
        graphon (np.ndarray): sometimes, initialize directly from graphon matrix,
        **args: additional args used in estimation method
    """
    def __init__(
        self, 
        graphs : list[np.ndarray], 
        padding : bool, 
        r : int, 
        align_max_size : int,
        label : np.ndarray,
        estimation_method : str = "usvt", 
        graphon: np.ndarray = None,
        **args):

        self.graphs = graphs 
        self.padding = padding 
        self.r = r 
        self.estimation_method = estimation_method
        self._label = label
        self.align_max_size = align_max_size
        self._graphon = graphon
            
        if (not graphon) and estimation_method in ['usvt']:
            self.estimate()
    
    def estimate(self):
        """Estimates the distribution
        
        Args:
            ...
        ...
        """
        align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(
                self.graphs[:self.align_max_size], padding=self.padding, N=self.r)
        graphon = largest_gap(align_graphs_list, k=self.r, sum_graph=sum_graph)
        np.fill_diagonal(graphon, 0)
        self._graphon = graphon 


    def generate(self, K): 
        """
        K (int): number of nodes in the new random graph
        Returns: newly generated graph (np.ndarray adjacency matrix) with K nodes
        """
        nodes = np.random.uniform(size=(K,))
        rounded_nodes = (nodes * self.r).astype(np.uint8)
        prob_vals = self._graphon[rounded_nodes[:, None], rounded_nodes]
        sampled_edges = (np.random.uniform(size=(K, K)) <= prob_vals)
        return sampled_edges 


    def mixup(self, graphon_matrix, label, la=0.5):
        self._label = la * self._label + (1 - la) * label 
        self._graphon = la * self._graphon + (1 - la) * graphon_matrix
         

    def get_graphon(self):
        return self._graphon 
