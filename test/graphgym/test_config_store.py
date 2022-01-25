import hydra
from omegaconf import DictConfig

import torch_geometric.graphgym.config_store  # noqa


def test_config_store():
    with hydra.initialize(config_path='.'):
        cfg = hydra.compose(config_name='my_config')

    assert len(cfg) == 1
    assert 'dataset' in cfg

    assert len(cfg.dataset) == 3
    assert cfg.dataset._target_.split('.')[-1] == 'KarateClub'

    assert isinstance(cfg.dataset.transform, DictConfig)
    assert len(cfg.dataset.transform) == 2
    assert 'NormalizeFeatures' in cfg.dataset.transform
    assert 'AddSelfLoops' in cfg.dataset.transform

    assert isinstance(cfg.dataset.pre_transform, DictConfig)
    assert len(cfg.dataset.pre_transform) == 0

    assert isinstance(cfg.dataset.transform.NormalizeFeatures, DictConfig)
    assert (cfg.dataset.transform.NormalizeFeatures._target_.split('.')[-1] ==
            'NormalizeFeatures')
    assert cfg.dataset.transform.NormalizeFeatures.attrs == ['x']

    assert isinstance(cfg.dataset.transform.AddSelfLoops, DictConfig)
    assert (cfg.dataset.transform.AddSelfLoops._target_.split('.')[-1] ==
            'AddSelfLoops')
    assert cfg.dataset.transform.AddSelfLoops.attr == 'edge_weight'
    assert cfg.dataset.transform.AddSelfLoops.fill_value is None
