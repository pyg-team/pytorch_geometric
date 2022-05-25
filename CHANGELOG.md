# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.0.5] - 2022-MM-DD
### Added
- Added `torch_geometric.nn.aggr` package ([#4687](https://github.com/pyg-team/pytorch_geometric/pull/4687))
- Added the `DimeNet++` model ([#4432](https://github.com/pyg-team/pytorch_geometric/pull/4432), [#4699](https://github.com/pyg-team/pytorch_geometric/pull/4699), [#4700](https://github.com/pyg-team/pytorch_geometric/pull/4700))
- Added an example of using PyG with PyTorch Ignite ([#4487](https://github.com/pyg-team/pytorch_geometric/pull/4487))
- Added `GroupAddRev` module with support for reducing training GPU memory ([#4671](https://github.com/pyg-team/pytorch_geometric/pull/4671), [#4701](https://github.com/pyg-team/pytorch_geometric/pull/4701), [#4715](https://github.com/pyg-team/pytorch_geometric/pull/4715))
- Added benchmarks via [`wandb`](https://wandb.ai/site) ([#4656](https://github.com/pyg-team/pytorch_geometric/pull/4656), [#4672](https://github.com/pyg-team/pytorch_geometric/pull/4672), [#4676](https://github.com/pyg-team/pytorch_geometric/pull/4676))
- Added `unbatch` functionality ([#4628](https://github.com/pyg-team/pytorch_geometric/pull/4628))
- Confirm that `to_hetero()` works with custom functions, *e.g.*, `dropout_adj` ([4653](https://github.com/pyg-team/pytorch_geometric/pull/4653))
- Added the `MLP.plain_last=False` option ([4652](https://github.com/pyg-team/pytorch_geometric/pull/4652))
- Added a check in `HeteroConv` and `to_hetero()` to ensure that `MessagePassing.add_self_loops` is disabled ([4647](https://github.com/pyg-team/pytorch_geometric/pull/4647))
- Added `HeteroData.subgraph()` support ([#4635](https://github.com/pyg-team/pytorch_geometric/pull/4635))
- Added the `AQSOL` dataset ([#4626](https://github.com/pyg-team/pytorch_geometric/pull/4626))
- Added `HeteroData.node_items()` and `HeteroData.edge_items()` functionality ([#4644](https://github.com/pyg-team/pytorch_geometric/pull/4644))
- Added PyTorch Lightning support in GraphGym ([#4531](https://github.com/pyg-team/pytorch_geometric/pull/4531))
- Added support for returning embeddings in `MLP` models ([#4625](https://github.com/pyg-team/pytorch_geometric/pull/4625))
- Added faster initialization of `NeighborLoader` in case edge indices are already sorted (via `is_sorted=True`) ([#4620](https://github.com/pyg-team/pytorch_geometric/pull/4620), [#4702](https://github.com/pyg-team/pytorch_geometric/pull/4702))
- Added `AddPositionalEncoding` transform ([#4521](https://github.com/pyg-team/pytorch_geometric/pull/4521))
- Added `HeteroData.is_undirected()` support ([#4604](https://github.com/pyg-team/pytorch_geometric/pull/4604))
- Added the `Genius` and `Wiki` datasets to `nn.datasets.LINKXDataset` ([#4570](https://github.com/pyg-team/pytorch_geometric/pull/4570), [#4600](https://github.com/pyg-team/pytorch_geometric/pull/4600))
- Added `nn.glob.GlobalPooling` module with support for multiple aggregations ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added support for graph-level outputs in `to_hetero` ([#4582](https://github.com/pyg-team/pytorch_geometric/pull/4582))
- Added `CHANGELOG.md` ([#4581](https://github.com/pyg-team/pytorch_geometric/pull/4581))
### Changed
- Fixed the ranking protocol bug in the RGCN link prediction example ([#4688](https://github.com/pyg-team/pytorch_geometric/pull/4688))
- Math support in Markdown ([#4683](https://github.com/pyg-team/pytorch_geometric/pull/4683))
- Allow for `setter` properties in `Data` ([#4682](https://github.com/pyg-team/pytorch_geometric/pull/4682), [#4686](https://github.com/pyg-team/pytorch_geometric/pull/4686))
- Allow for optional `edge_weight` in `GCN2Conv` ([#4670](https://github.com/pyg-team/pytorch_geometric/pull/4670))
- Fixed the interplay between `TUDataset` and `pre_transform` that modify node features ([#4669](https://github.com/pyg-team/pytorch_geometric/pull/4669))
- Make use of the `pyg_sphinx_theme` documentation template ([#4664](https://github.com/pyg-team/pyg-lib/pull/4664), [#4667](https://github.com/pyg-team/pyg-lib/pull/4667))
- Refactored reading molecular positions from sdf file for qm9 datasets ([4654](https://github.com/pyg-team/pytorch_geometric/pull/4654))
- Fixed `MLP.jittable()` bug in case `return_emb=True` ([#4645](https://github.com/pyg-team/pytorch_geometric/pull/4645), [#4648](https://github.com/pyg-team/pytorch_geometric/pull/4648))
- The generated node features of `StochasticBlockModelDataset` are now ordered with respect to their labels ([#4617](https://github.com/pyg-team/pytorch_geometric/pull/4617))
- Removed unnecessary colons and fixed typos in the documentation ([#4616](https://github.com/pyg-team/pytorch_geometric/pull/4616))
- The `bias` argument in `TAGConv` is now actually applied ([#4597](https://github.com/pyg-team/pytorch_geometric/pull/4597))
- Fixed subclass behaviour of `process` and `download` in `Datsaet` ([#4586](https://github.com/pyg-team/pytorch_geometric/pull/4586))
- Fixed filtering of attributes for loaders in case `__cat_dim__ != 0` ([#4629](https://github.com/pyg-team/pytorch_geometric/pull/4629))
### Removed
