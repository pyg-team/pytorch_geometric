# Graph Classification

Evaluation script for various methods on [common benchmark datasets](http://graphkernels.cs.tu-dortmund.de) via 10-fold cross validation, where a training fold is randomly sampled to serve as a validation set.
Hyperparameter selection is performed for the number of hidden units and the number of layers with respect to the validation set:

* **[GCN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gcn.py)**
* **[GraphSAGE](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py)**
* **[GIN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py)**
* **[Graclus](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/graclus.py)**
* **[Top-K Pooling](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/top_k.py)**
* **[SAG Pooling](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sag_pool.py)**
* **[DiffPool](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/diff_pool.py)**
* **[EdgePool](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/edge_pool.py)**
* **[GlobalAttention](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/global_attention.py)**
* **[Set2Set](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py)**
* **[SortPool](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sort_pool.py)**
* **[ASAPool](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/asap.py)**

Run (or modify) the whole test suite via

```
$ python main.py
```

For more comprehensive time-measurement and memory usage information, you may use

```
$ python main_performance.py
```
