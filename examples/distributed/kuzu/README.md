# Using Kùzu as a Remote Backend for PyG

[Kùzu](https://kuzudb.com/) is an in-process property graph database management system built for query speed and scalability.
It provides an integration with PyG via the [remote backend interface](https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html) of PyG.
The Python API of Kùzu outputs a [`torch_geometric.data.FeatureStore`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.FeatureStore.html) and a [`torch_geometric.data.GraphStore`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.GraphStore.html) that can be plugged directly into existing familiar PyG interfaces such as [`NeighborLoader`](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html) and enables training GNNs directly on graphs stored in Kùzu.
This is particularly useful if you would like to train graphs that don't fit on your CPU's memory.

## Installation

You can install Kùzu as follows:

```bash
pip install kuzu
```

## Usage

The API and design documentation of Kùzu can be found at [https://kuzudb.com/docs/](https://kuzudb.com/docs/).

## Examples

We provide the following examples to showcase the usage of Kùzu remote backend within PyG:

### PubMed

<a target="_blank" href="https://colab.research.google.com/drive/12fOSqPm1HQTz_m9caRW7E_92vaeD9xq6">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The PubMed example is hosted on [Google Colab](https://colab.research.google.com/drive/12fOSqPm1HQTz_m9caRW7E_92vaeD9xq6).
In this example, we work on a small dataset for demonstrative purposes.
The [PubMed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html) dataset consists of 19,717 papers as nodes and 88,648 citation relationships between them.

### `papers_100M`

This example shows how to use the remote backend feature of Kùzu to work with a large graph of papers and citations on a single machine.
The data used in this example is `ogbn-papers100M` from the [Open Graph Benchmark](https://ogb.stanford.edu/).
The dataset contains approximately 111 million nodes and 1.6 billion edges.
