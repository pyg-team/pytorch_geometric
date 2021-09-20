# Semi-supervised Node Classification

Evaluation scripts for various methods on the Cora, CiteSeer and PubMed citation networks.
Each experiment is repeated 100 times on either a fixed train/val/test split or on multiple random splits:

* **[GCN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/gcn.py)**: `python gcn.py`
* **[GAT](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/gat.py)**: `python gat.py`
* **[Cheby](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/cheb.py)**: `python cheb.py`
* **[SGC](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/sgc.py)**: `python sgc.py`
* **[ARMA](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/arma.py)**: `python arma.py`
* **[APPNP](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/appnp.py)**: `python appnp.py`

Run the whole test suite via

```
$ ./run.sh
```
