`*.to_hetero` provides an elegant way to **convert any homogeneuous GNN model into its corresponding heterogeneous variant**:

```python
model = Sequential(', edge_index', [
    (LazyGCNConv(16), 'x, edge_index -> x'),
    ReLU(),
    (LazyGCNConv(16), 'x, edge_index -> x'),
    ReLU(),
])

node_types = ['paper', 'author']
edge_types = [('paper', 'paper'), ('author', 'paper'), ('paper', 'author')]

model = model.to_hetero((node_types, edge_types), aggr='sum')
```

## Motivation

Writing heterogeneous models is non-trivial and cumbersome since one needs to keep track of the state of various node and edge types.

## Features

`*.to_hetero` provides full support for:

* **Lazy initialization** of GNN operators
* Converting GNN operators into **bipartite instances**
* **Various heterogeneous GNN variants**: distinct parameters, shared parameters, basis decomposition
* **Sophisticated GNN models**, *e.g.*, residual connections or jumping knowledge
