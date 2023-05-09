# Using Kùzu as a remote backend for PyG Examples

[Kùzu](https://kuzudb.com/) is an in-process property graph database management system built for query speed and scalability.
It provides an integration with PyG via the [remote backend interface](https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html) of PyG.
The Python API of Kùzu can output `torch_geometric.data.FeatureStore` and a `torch_geometric.data.GraphStore` objects that plug directly into existing familiar PyG interfaces such as [`neighbor_loader`](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html) and enables training GNNs directly on graph stored in Kùzu.
This is particularly useful if you would like to train graphs that don't fit on your CPU's memory.

## Installation

You can install Kùzu as follows:

```bash
pip install kuzu
```

## Usage

The API and design documentation of Kùzu can be found at [https://kuzudb.com/docs/](https://kuzudb.com/docs/).

## Examples

We provide two examples to showcase the usage of Kùzu remote backend within PyG:

### PubMed example

<a target="_blank" href="https://colab.research.google.com/drive/12fOSqPm1HQTz_m9caRW7E_92vaeD9xq6">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The PubMed example is hosted on [Google Colab](https://colab.research.google.com/drive/12fOSqPm1HQTz_m9caRW7E_92vaeD9xq6).
In this example, we work on a small dataset for demonstrative purposes.
The dataset "PubMed" from the “Revisiting Semi-Supervised Learning with Graph Embeddings” paper.
This is one of the [benchmark datasets in PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch-geometric-datasets-planetoid).
The PubMed dataset consists of 19,717 papers as nodes and 88,648 citation relationships between them.
Each paper has a feature vector (`x`) with a dimension of 500, and a category (`y`).

### papers_100M example

This example shows how to use the remote backend feature of Kùzu to work with a large graph of papers and citations on a single machine.
The data used in this example is `ogbn-papers100M` from the [Open Graph Benchmark](https://ogb.stanford.edu/).
The dataset contains approximately 111 million nodes and 1.6 billion edges.
