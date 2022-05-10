# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.0.5] - 2022-MM-DD
### Added
- Added `HeteroData.is_undirected()` support ([#4604](https://github.com/pyg-team/pytorch_geometric/pull/4604))
- Added the `Genius` and `Wiki` datasets to `nn.datasets.LINKXDataset` ([#4570](https://github.com/pyg-team/pytorch_geometric/pull/4570), [#4600](https://github.com/pyg-team/pytorch_geometric/pull/4600))
- Added `nn.glob.GlobalPooling` module with support for multiple aggregations ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added support for graph-level outputs in `to_hetero` ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added `CHANGELOG.md` ([#4581](https://github.com/pyg-team/pytorch_geometric/pull/4581))
### Changed
- The `bias` argument in `TAGConv` is now actually applied ([#4597](https://github.com/pyg-team/pytorch_geometric/pull/4597))
- Fixed subclass behaviour of `process` and `download` in `Datsaet` ([#4586](https://github.com/pyg-team/pytorch_geometric/pull/4586))
### Removed
