# Papers-100M example

This example shows how to use the remote backend feature of [Kùzu](https://kuzudb.com) to work with a large graph of papers and citations on a single machine.
The data used in this example is `ogbn-papers100M` from the [Open Graph Benchmark](https://ogb.stanford.edu/).
The dataset contains approximately 100 million nodes and 2.5 billion edges.

## Install the dependencies

```bash
pip install pandas kuzu tqdm
```

## Prepare the data

1. Download the dataset from [http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip](http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip).
   Put the zip file into this directory.
2. Run `prepare_data.py`.
   The script will automatically extract the data and convert to the format that Kùzu can read.
   A Kùzu database instance is then created under `papers100M` and the data is loaded into the it.

## Train a model

Run `train.py`.
The script will train a GNN model on the dataset.
The model is a 3-layer GraphSAGE model ported from [https://github.com/mengyangniu/ogbn-papers100m-sage/](https://github.com/mengyangniu/ogbn-papers100m-sage/).
