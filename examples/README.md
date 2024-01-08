This folder contains a plethora of examples covering nearly any possible GNN use case. This readme highlights some key examples.

A great simple example to start with is `gcn.py`, showing a user how to train a GCN for node level prediction on small scale homogeneous data.

For a simple link prediction example see `link_pred.py`.

For examples training on [Open Graph Benchmark](https://ogb.stanford.edu/) dataset, see `ogbn_*.py`:
	- `ogbn_products_gat.py` and `ogbn_products_sage.py` show how to train GAT or GraphSage models on the OGBN Products dataset.
	- `ogbn_protiens_deepgcn.py` is a great example to see how to train a DeepGCN on the OGBN Protiens dataset
	- `ogbn_papers_100m.py` is a great example for training a GCN on OGBN Papers100m. This dataset is much larger than the others with ~1.6B edges.

For examples on using torch.compile see the examples under examples/compile.

For examples on scaling up to using multiple GPUs see the README under examples/multi_gpu.

For examples on working with heterogeneous data, see the README under examples/hetero.