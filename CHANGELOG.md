# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.0.5] - 2022-MM-DD
### Added
- Added PyTorch Lightning support in GraphGym ([#4531](https://github.com/pyg-team/pytorch_geometric/pull/4531))
- Added support for returning embeddings in `MLP` models ([#4625](https://github.com/pyg-team/pytorch_geometric/pull/4625))
- Added faster initialization of `NeighborLoader` in case edge indices are already sorted (via `is_sorted=True`) ([#4620](https://github.com/pyg-team/pytorch_geometric/pull/4620))
- Added `AddPositionalEncoding` transform ([#4521](https://github.com/pyg-team/pytorch_geometric/pull/4521))
- Added `HeteroData.is_undirected()` support ([#4604](https://github.com/pyg-team/pytorch_geometric/pull/4604))
- Added the `Genius` and `Wiki` datasets to `nn.datasets.LINKXDataset` ([#4570](https://github.com/pyg-team/pytorch_geometric/pull/4570), [#4600](https://github.com/pyg-team/pytorch_geometric/pull/4600))
- Added `nn.glob.GlobalPooling` module with support for multiple aggregations ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added support for graph-level outputs in `to_hetero` ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added `CHANGELOG.md` ([#4581](https://github.com/pyg-team/pytorch_geometric/pull/4581))
### Changed
- The generated node features of `StochasticBlockModelDataset` are now ordered with respect to their labels ([#4617](https://github.com/pyg-team/pytorch_geometric/pull/4617))
- Removed unnecessary colons and fixed typos in the documentation ([#4616](https://github.com/pyg-team/pytorch_geometric/pull/4616))
- The `bias` argument in `TAGConv` is now actually applied ([#4597](https://github.com/pyg-team/pytorch_geometric/pull/4597))
- Fixed subclass behaviour of `process` and `download` in `Datsaet` ([#4586](https://github.com/pyg-team/pytorch_geometric/pull/4586))
- Fixed filtering of attributes for loaders in case `__cat_dim__ != 0` ([#4629](https://github.com/pyg-team/pytorch_geometric/pull/4629))
### Removed
