import os
import os.path as osp
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import subgraph


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore
    with open(path) as f:
        return yaml.safe_load(f)


class GraphLandDataset(InMemoryDataset):
    r"""The graph datasets from the `"GraphLand: Evaluating
    Graph Machine Learning Models on Diverse Industrial Data"
    <https://arxiv.org/abs/2409.14500>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"hm-categories"`,
            :obj:`"pokec-regions"`, :obj:`"web-topics"`, :obj:`"tolokers-2"`,
            :obj:`"city-reviews"`, :obj:`"artnet-exp"`, :obj:`"web-fraud"`,
            :obj:`"hm-prices"`, :obj:`"avazu-ctr"`, :obj:`"city-roads-M"`,
            :obj:`"city-roads-L"`, :obj:`"twitch-views"`,
            :obj:`"artnet-views"`, :obj:`"web-traffic"`).
        split (str): The type of dataset split/setting (:obj:`"RL"`,
            :obj:`"RH"`, :obj:`"TH"`, :obj:`"THI"`).
            :obj:`"RL"` is for "random low" split — a 10%/10%/80% random
            stratified train/val/test split.
            :obj:`"RH"` is for "random high" split — a 50%/25%/25% random
            stratified train/val/test split.
            :obj:`"TH"` is for "temporal high" split — a 50%/25%/25% temporal
            train/val/test split.
            :obj:`"THI"` is for "temporal high" split with the inductive
            setting, which means that val and test nodes are not seen at train
            time, and test nodes are not seen at val time. In contrast to the
            previous three splits that will result in a dataset with a single
            graph, setting the split to :obj:`"THI"` will result in a dataset
            with three graphs corresponding to the train, val, and test
            snapshots of an evolving network.
            :obj:`"TH"` and :obj:`"THI"` splits are not available for the
            following datasets: :obj:`"city-reviews"`, :obj:`"city-roads-M"`,
            :obj:`"city-roads-L"`, :obj:`"web-traffic"`.
        numerical_features_transform (str, optional): A transform applied to
            numerical features (:obj:`None`, :obj:`"standard_scaler"`,
            :obj:`"min_max_scaler"`, :obj:`"quantile_transform_normal"`,
            :obj:`"quantile_transform_uniform"`). Since numerical features can
            have widely different scales and distributions, it is typically
            useful to apply some transform to them before passing them to a
            neural model. This transform is applied to all numerical features
            except for those that are also categorized as fraction features.
            (default :obj:`"quantile_transform_normal"`)
        fraction_features_transform (str, optional): A transform applied to
            fraction features (:obj:`None`, :obj:`"standard_scaler"`,
            :obj:`"min_max_scaler"`, :obj:`"quantile_transform_normal"`,
            :obj:`"quantile_transform_uniform"`). Fraction features are a
            subset of numerical features that have the meaning of fractions
            and are thus always in :obj:`[0, 1]` range. Since their range is
            bounded, it is not neccessary but may still be useful to apply
            some transform to them before passing them to a neural model.
            (default :obj:`None`)
        categorical_features_transform (str, optional): A transform applied to
            categorical features (:obj:`None`, :obj:`"one_hot_encoding"`).
            It is most often useful to apply one-hot encoding to categorical
            features before passing them to a neural model.
            (default :obj:`one_hot_encoding`)
        regression_targets_transform (str, optional): A transform applied to
            regression targets (:obj:`None`, :obj:`"standard_scaler"`,
            :obj:`"min_max_scaler"`). Depending on their range, it may or may
            not be useful to apply a transform to regression targets before
            fitting a neural model to them. This argument does not affect
            classification datasets. (default :obj:`"standard_scaler"`)
        numerical_features_nan_imputation_strategy (str, optional): Defines
            which value to fill NaNs in numerical features with
            (:obj:`None`, :obj:`"mean"`, :obj:`"median"`,
            :obj:`"most_frequent"`). This imputation strategy is applied to
            all numerical features except for those that are also categorized
            as fraction features. (default :obj:`"most_frequent"`)
        fraction_features_nan_imputation_strategy (str, optional): Defines
            which value to fill NaNs in fraction features with (:obj:`None`,
            :obj:`"mean"`, :obj:`"median"`, :obj:`"most_frequent"`).
            (default :obj:`"most_frequent"`)
        to_undirected (bool, optional): Whether to convert a directed graph
            to an undirected one. Does not affect undirected graphs.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 14 10 10 10 14
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - is directed
          - task
        * - :obj:`hm-categories`
          - 46,563
          - 21,461,990
          - False
          - multiclass
        * - :obj:`pokec-regions`
          - 1,632,803
          - 30,622,564
          - True
          - multiclass
        * - :obj:`web-topics`
          - 2,890,331
          - 12,895,369
          - True
          - multiclass
        * - :obj:`tolokers-2`
          - 11,758
          - 1,038,000
          - False
          - binclass
        * - :obj:`city-reviews`
          - 148,801
          - 2,330,830
          - False
          - binclass
        * - :obj:`artnet-exp`
          - 50,405
          - 560,696
          - False
          - binclass
        * - :obj:`web-fraud`
          - 2,890,331
          - 12,895,369
          - True
          - binclass
        * - :obj:`hm-prices`
          - 46,563
          - 21,461,990
          - False
          - regression
        * - :obj:`avazu-ctr`
          - 76,269
          - 21,968,154
          - False
          - regression
        * - :obj:`city-roads-M`
          - 57,073
          - 132,571
          - True
          - regression
        * - :obj:`city-roads-L`
          - 142,257
          - 279,062
          - True
          - regression
        * - :obj:`twitch-views`
          - 168,114
          - 13,595,114
          - False
          - regression
        * - :obj:`artnet-views`
          - 50,405
          - 560,696
          - False
          - regression
        * - :obj:`web-traffic`
          - 2,890,331
          - 12,895,369
          - True
          - regression
    """
    _url = 'https://zenodo.org/records/16895532'
    GRAPHLAND_DATASETS = {
        'hm-categories': 'multiclass_classification',
        'pokec-regions': 'multiclass_classification',
        'web-topics': 'multiclass_classification',
        'tolokers-2': 'binary_classification',
        'city-reviews': 'binary_classification',
        'artnet-exp': 'binary_classification',
        'web-fraud': 'binary_classification',
        'hm-prices': 'regression',
        'avazu-ctr': 'regression',
        'city-roads-M': 'regression',
        'city-roads-L': 'regression',
        'twitch-views': 'regression',
        'artnet-views': 'regression',
        'web-traffic': 'regression',
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
        numerical_features_transform: Optional[str] = 'default',
        fraction_features_transform: Optional[str] = 'default',
        categorical_features_transform: Optional[str] = 'one_hot_encoding',
        regression_targets_transform: Optional[str] = 'default',
        numerical_features_nan_imputation_strategy: Optional[
            str] = 'most_frequent',
        fraction_features_nan_imputation_strategy: Optional[
            str] = 'most_frequent',
        to_undirected: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in self.GRAPHLAND_DATASETS, (
            f'Unsupported dataset name: {name}')

        assert split in ['RL', 'RH', 'TH', 'THI'], \
            f'Unsupported split name: {split}'
        if split in ['TH', 'THI']:
            assert name not in [
                'city-reviews',
                'city-roads-M',
                'city-roads-L',
                'web-trafic',
            ], ('Temporal split is not available for city-reviews, '
                'city-roads-M, city-roads-L, web-trafic.')

        if numerical_features_transform == 'default':
            numerical_features_transform = 'quantile_transform_normal'

        if fraction_features_transform == 'default':
            fraction_features_transform = (  #
                'quantile_transform_normal'
                if name in ['artnet-views', 'avazu-ctr'] else None)

        if regression_targets_transform == 'default':
            regression_targets_transform = 'standard_scaler'

        if numerical_features_transform is not None:
            assert numerical_features_transform in [
                'standard_scaler',
                'min_max_scaler',
                'quantile_transform_normal',
                'quantile_transform_uniform',
            ], ('Unsupported numerical features transform: '
                f'{numerical_features_transform}')

        if fraction_features_transform is not None:
            assert fraction_features_transform in [
                'standard_scaler',
                'min_max_scaler',
                'quantile_transform_normal',
                'quantile_transform_uniform',
            ], ('Unsupported fraction features transform: '
                f'{fraction_features_transform}')

        if categorical_features_transform is not None:
            assert categorical_features_transform == 'one_hot_encoding', (
                'Unsupported categorical features transform: '
                f'{categorical_features_transform}')

        if regression_targets_transform is not None:
            assert regression_targets_transform in [
                'standard_scaler', 'min_max_scaler'
            ], ('Unsupported regression targets transform:'
                f'{regression_targets_transform}')

        self.name = name
        self.split = split
        self.task = self.GRAPHLAND_DATASETS[name]
        self._num_transform = numerical_features_transform
        self._frac_transform = fraction_features_transform
        self._cat_transform = categorical_features_transform
        self._reg_transform = regression_targets_transform
        self._num_imputation = numerical_features_nan_imputation_strategy
        self._frac_imputation = fraction_features_nan_imputation_strategy
        self._to_undirected = to_undirected

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            MinMaxScaler,
            OneHotEncoder,
            QuantileTransformer,
            StandardScaler,
        )
        self._transforms = {
            'standard_scaler':
            partial(StandardScaler, copy=False),
            'min_max_scaler':
            partial(MinMaxScaler, clip=False, copy=False),
            'quantile_transform_normal':
            partial(
                QuantileTransformer,
                output_distribution='normal',
                subsample=None,
                random_state=0,
                copy=False,
            ),
            'quantile_transform_uniform':
            partial(
                QuantileTransformer,
                output_distribution='uniform',
                subsample=None,
                random_state=0,
                copy=False,
            ),
            'one_hot_encoding':
            partial(
                OneHotEncoder,
                drop='if_binary',
                sparse_output=False,
                handle_unknown='ignore',
                dtype=np.float32,
            ),
        }
        self._imputer = SimpleImputer

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        specs = ''.join(f'__{str(arg).lower()}' for arg in [
            self.split,
            self._num_transform,
            self._frac_transform,
            self._cat_transform,
            self._reg_transform,
            self._num_imputation,
            self._frac_imputation,
            self._to_undirected,
        ])
        return osp.join(self.root, self.name, 'processed', specs)

    @property
    def raw_file_names(self) -> str:
        return self.name

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        zip_url = osp.join(self._url, 'files', f'{self.name}.zip')
        path = download_url(zip_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def _get_raw_data(self) -> dict:
        import pandas as pd

        raw_data_dir = osp.join(self.raw_dir, self.name)
        info = _load_yaml(osp.join(raw_data_dir, 'info.yaml'))

        features_df = pd.read_csv(
            osp.join(raw_data_dir, 'features.csv'),
            index_col=0,
        )
        num_features_names = [
            name for name in info['numerical_features_names']
            if name not in info['fraction_features_names']
        ]
        num_features = features_df[num_features_names].values
        num_features = num_features.astype(np.float32)

        cat_features_names = info['categorical_features_names']
        cat_features = features_df[cat_features_names].values
        cat_features = cat_features.astype(np.int32)

        frac_features_names = info['fraction_features_names']
        frac_features = features_df[frac_features_names].values
        frac_features = frac_features.astype(np.float32)

        targets_df = pd.read_csv(
            osp.join(raw_data_dir, 'targets.csv'),
            index_col=0,
        )
        targets = targets_df[info['target_name']].values
        targets = targets.astype(np.float32)

        masks_df = pd.read_csv(
            osp.join(raw_data_dir, f'split_masks_{self.split[:2]}.csv'),
            index_col=0,
        )
        masks = {
            k: np.array(v, dtype=bool)
            for k, v in masks_df.to_dict('list').items()
        }

        edges_df = pd.read_csv(osp.join(raw_data_dir, 'edgelist.csv'))
        edges = edges_df.values

        return {
            'info': info,
            'num_features': num_features,
            'cat_features': cat_features,
            'frac_features': frac_features,
            'targets': targets,
            'masks': masks,
            'edges': edges,
        }

    def _get_transductive_data(self) -> list[Data]:
        raw_data = self._get_raw_data()

        # >>> process targets
        targets = raw_data['targets']
        labeled_mask = ~np.isnan(targets)
        if (raw_data['info']['task'] == 'regression'
                and self._reg_transform is not None):
            targets = targets.reshape(-1, 1)
            transform = self._transforms[self._reg_transform]()
            transform.fit(targets[raw_data['masks']['train']])
            targets = transform.transform(targets).reshape(-1)
        targets = torch.from_numpy(targets).float()

        # >>> process numerical features
        num_features = raw_data['num_features']
        if num_features.size > 0:
            if self._num_transform is not None:
                transform = self._transforms[self._num_transform]()
                transform.fit(num_features)

            num_features = self._imputer(
                missing_values=np.nan, strategy=self._num_imputation,
                copy=False).fit_transform(num_features)

            if self._num_transform is not None:
                num_features = transform.transform(num_features)

        # >>> process fraction features
        frac_features = raw_data['frac_features']
        if frac_features.size > 0:
            if self._frac_transform is not None:
                transform = self._transforms[self._frac_transform]()
                transform.fit(frac_features)

            frac_features = self._imputer(
                missing_values=np.nan, strategy=self._frac_imputation,
                copy=False).fit_transform(frac_features)

            if self._frac_transform is not None:
                frac_features = transform.transform(frac_features)

        # >>> process categorical features
        cat_features = raw_data['cat_features']
        if cat_features.size > 0 and self._cat_transform is not None:
            cat_features = (self._transforms[self._cat_transform]
                            ().fit_transform(cat_features))

        # >>> concatenate features and make features mask
        features = np.concatenate(
            [num_features, frac_features, cat_features],
            axis=1,
        )
        features = torch.from_numpy(features).float()

        num_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        num_mask[:num_features.shape[1]] = True

        frac_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        frac_mask[num_features.shape[1]:-cat_features.shape[1]] = True

        cat_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        cat_mask[-cat_features.shape[1]:] = True

        # >>> update split masks
        train_mask = raw_data['masks']['train'] & labeled_mask
        train_mask = torch.from_numpy(train_mask).bool()

        val_mask = raw_data['masks']['val'] & labeled_mask
        val_mask = torch.from_numpy(val_mask).bool()

        test_mask = raw_data['masks']['test'] & labeled_mask
        test_mask = torch.from_numpy(test_mask).bool()

        # >>> make edge index
        edge_index = raw_data['edges'].T
        edge_index = torch.from_numpy(edge_index).long()

        # >>> construct Data object
        data = Data(
            edge_index=edge_index,
            x=features,
            y=targets,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            x_numerical_mask=num_mask,
            x_fraction_mask=frac_mask,
            x_categorical_mask=cat_mask,
        )
        return [data]

    def _get_inductive_data(self) -> list[Data]:
        raw_data = self._get_raw_data()
        transform_mask = raw_data['masks']['train']

        # >>> process targets
        targets = raw_data['targets']
        labeled_mask = ~np.isnan(targets)
        if (raw_data['info']['task'] == 'regression'
                and self._reg_transform is not None):
            targets = targets.reshape(-1, 1)
            transform = self._transforms[self._reg_transform]()
            transform.fit(targets[transform_mask])
            targets = transform.transform(targets).reshape(-1)
        targets = torch.from_numpy(targets).float()

        # >>> process numerical features
        num_features = raw_data['num_features']
        if num_features.size > 0:
            if self._num_transform is not None:
                transform = self._transforms[self._num_transform]()
                transform.fit(num_features[transform_mask])

            imputer = self._imputer(missing_values=np.nan,
                                    strategy=self._num_imputation, copy=False)
            imputer.fit(num_features[transform_mask])
            num_features = imputer.transform(num_features)

            if self._num_transform is not None:
                num_features = transform.transform(num_features)

        # >>> process fraction features
        frac_features = raw_data['frac_features']
        if frac_features.size > 0:
            if self._frac_transform is not None:
                transform = self._transforms[self._frac_transform]()
                transform.fit(frac_features[transform_mask])

            imputer = self._imputer(missing_values=np.nan,
                                    strategy=self._frac_imputation, copy=False)
            imputer.fit(frac_features[transform_mask])
            frac_features = imputer.transform(frac_features)

            if self._frac_transform is not None:
                frac_features = transform.transform(frac_features)

        # >>> process categorical features
        cat_features = raw_data['cat_features']
        if cat_features.size > 0 and self._cat_transform is not None:
            transform = self._transforms[self._cat_transform]()
            transform.fit(cat_features[transform_mask])
            cat_features = transform.transform(cat_features)

        # >>> concatenate features and make features mask
        features = np.concatenate(
            [num_features, frac_features, cat_features],
            axis=1,
        )
        features = torch.from_numpy(features).float()

        num_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        num_mask[:num_features.shape[1]] = True

        frac_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        frac_mask[num_features.shape[1]:-cat_features.shape[1]] = True

        cat_mask = torch.zeros(features.shape[1], dtype=torch.bool)
        cat_mask[-cat_features.shape[1]:] = True

        # >>> construct Data objects
        edge_index = raw_data['edges'].T
        edge_index = torch.from_numpy(edge_index).long()

        # --- train
        train_graph_mask = raw_data['masks']['train']
        train_graph_mask = torch.from_numpy(train_graph_mask).bool()

        train_label_mask = raw_data['masks']['train'] & labeled_mask
        train_label_mask = torch.from_numpy(train_label_mask).bool()

        train_node_id = np.where(train_graph_mask)[0]
        train_node_id = torch.from_numpy(train_node_id).bool()  # type: ignore

        train_edge_index, _ = subgraph(
            train_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        train_data = Data(
            edge_index=train_edge_index,
            x=features[train_graph_mask],
            y=targets[train_graph_mask],
            mask=train_label_mask[train_graph_mask],
            x_numerical_mask=num_mask,
            x_fraction_mask=frac_mask,
            x_categorical_mask=cat_mask,
            node_id=train_node_id,
        )

        # --- val
        val_graph_mask = (raw_data['masks']['train']
                          | raw_data['masks']['val'])
        val_graph_mask = torch.from_numpy(val_graph_mask).bool()

        val_label_mask = raw_data['masks']['val'] & labeled_mask
        val_label_mask = torch.from_numpy(val_label_mask).bool()

        val_node_id = np.where(val_graph_mask)[0]
        val_node_id = torch.from_numpy(val_node_id).long()  # type: ignore

        val_edge_index, _ = subgraph(
            val_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        val_data = Data(
            edge_index=val_edge_index,
            x=features[val_graph_mask],
            y=targets[val_graph_mask],
            mask=val_label_mask[val_graph_mask],
            x_numerical_mask=num_mask,
            x_fraction_mask=frac_mask,
            x_categorical_mask=cat_mask,
            node_id=val_node_id,
        )

        # --- test
        test_graph_mask = (raw_data['masks']['train']
                           | raw_data['masks']['val']
                           | raw_data['masks']['test'])
        test_graph_mask = torch.from_numpy(test_graph_mask).bool()

        test_label_mask = raw_data['masks']['test'] & labeled_mask
        test_label_mask = torch.from_numpy(test_label_mask).bool()

        test_node_id = np.where(test_graph_mask)[0]
        test_node_id = torch.from_numpy(test_node_id).long()  # type: ignore

        test_edge_index, _ = subgraph(
            test_graph_mask,
            edge_index,
            relabel_nodes=True,
        )
        test_data = Data(
            edge_index=test_edge_index,
            x=features[test_graph_mask],
            y=targets[test_graph_mask],
            mask=test_label_mask[test_graph_mask],
            x_numerical_mask=num_mask,
            x_fraction_mask=frac_mask,
            x_categorical_mask=cat_mask,
            node_id=test_node_id,
        )

        return [train_data, val_data, test_data]

    def process(self) -> None:
        data = (self._get_transductive_data() if self.split
                in ['RL', 'RH', 'TH'] else self._get_inductive_data())
        if self._to_undirected:
            transform = ToUndirected()
            for idx, d in enumerate(data):
                data[idx] = transform(d)

        self.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
