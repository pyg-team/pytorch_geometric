README for dataset MUTAG


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DD_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

The MUTAG dataset consists of 188 chemical compounds divided into two 
classes according to their mutagenic effect on a bacterium. 

The chemical data was obtained form http://cdb.ics.uci.edu and converted 
to graphs, where vertices represent atoms and edges represent chemical 
bonds. Explicit hydrogen atoms have been removed and vertices are labeled
by atom type and edges by bond type (single, double, triple or aromatic).
Chemical data was processed using the Chemistry Development Kit (v1.4).

Node labels:

  0  C
  1  N
  2  O
  3  F
  4  I
  5  Cl
  6  Br

Edge labels:

  0  aromatic
  1  single
  2  double
  3  triple


=== Previous Use of the Dataset ===

Kriege, N., Mutzel, P.: Subgraph matching kernels for attributed graphs. In: Proceedings
of the 29th International Conference on Machine Learning (ICML-2012) (2012).


=== References ===

Debnath, A.K., Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., and Hansch, C.
Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.
Correlation with molecular orbital energies and hydrophobicity. J. Med. Chem. 34(2):786-797 (1991).
