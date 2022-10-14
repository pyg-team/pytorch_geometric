import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
from tqdm import tqdm
import torch_geometric
import networkx as nx


def get_k_hop_adjacencies(edge_index, max_k, stack_edge_indices=False):
  """Return list of matrices/edge indices for 1,..,k-hop adjacency matrices
  n.b. binary matrix
  n.b. pretty inefficient"""
  tmp = to_dense_adj(edge_index).float()
  adj = tmp.to_sparse().float()
  idxs, matrices = [edge_index], [tmp]
  cutoffs, n_edges_per_k = [0], edge_index.shape[-1]
  for k in range(2, max_k+1):
    tmp = torch.bmm(adj, tmp)
    for i in range(tmp.shape[-1]):
      tmp[0, i, i] = 0 # remove self-connections
    tmp = (tmp>0).float() # remove edge multiples
    for m in matrices:
      tmp -= m
    tmp = (tmp>0).float() # remove -ves, cancelled edges
    idx, _ = dense_to_sparse(tmp) # outputs int64, which we want
    matrices.append(tmp)
    idxs.append(idx)
    cutoffs.append(n_edges_per_k)
    n_edges_per_k += idx.shape[-1]
    if torch.sum(tmp) == 0:
      break # adj matrix is empty
  cutoffs.append(n_edges_per_k)
  if stack_edge_indices:
    idxs = torch.cat(idxs, dim=-1)
  # matrices = torch.stack(matrices, dim=1)
  return idxs, get_khop_labels(cutoffs)


def get_khop_labels(cutoffs):
  # generates k-hop edge labels from cutoff tensor - used when all k-hop indices are put in Data.edge_index
  num_per_k = [cutoffs[i+1]-cutoffs[i] for i in range(len(cutoffs)-1)]
  edge_khop_labels = []
  for k in range(1, len(cutoffs)):
    edge_khop_labels.append(k*torch.ones(num_per_k[k-1]))
  return torch.cat(edge_khop_labels).reshape((-1, 1))


def plot_topology(edge_index):
  """Plot topology of graph"""
  # edge_index = torch.tensor([[0, 1, 1, 2],
  #                           [1, 0, 2, 1]], dtype=torch.long)
  x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
  data = torch_geometric.data.Data(x=x, edge_index=edge_index)
  g = torch_geometric.utils.to_networkx(data, to_undirected=True)
  nx.draw(g, with_labels=True)

def make_k_hop_edge_dataset(dataset):
  """Append k-hop edges to edge_index and set edge_attrs with k-hop labels"""
  # DO STUFF
  return dataset
