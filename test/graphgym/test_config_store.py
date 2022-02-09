import hydra
from omegaconf import DictConfig

import torch_geometric.graphgym.config_store  # noqa


def test_config_store():
    with hydra.initialize(config_path='.'):
        cfg = hydra.compose(config_name='my_config')

    assert len(cfg) == 4
    assert 'dataset' in cfg
    assert 'model' in cfg
    assert 'optim' in cfg
    assert 'scheduler' in cfg

    # Check `cfg.dataset`:
    assert len(cfg.dataset) == 3
    assert cfg.dataset._target_.split('.')[-1] == 'KarateClub'

    # Check `cfg.dataset.transform`:
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

    # Check `cfg.model`:
    assert len(cfg.model) == 11
    assert cfg.model._target_.split('.')[-1] == 'GCN'
    assert cfg.model.in_channels == 34
    assert cfg.model.out_channels == 4
    assert cfg.model.hidden_channels == 16
    assert cfg.model.num_layers == 2
    assert cfg.model.dropout == 0.0
    assert cfg.model.act == 'relu'
    assert cfg.model.norm is None
    assert cfg.model.jk is None
    assert not cfg.model.act_first
    assert cfg.model.act_kwargs is None

    # Check `cfg.optim`:
    assert len(cfg.optim) == 6
    assert cfg.optim._target_.split('.')[-1] == 'Adam'
    assert cfg.optim.lr == 0.001
    assert cfg.optim.betas == [0.9, 0.999]
    assert cfg.optim.eps == 1e-08
    assert cfg.optim.weight_decay == 0
    assert not cfg.optim.amsgrad

    # Check `cfg.scheduler`:
    assert len(cfg.scheduler) == 10
    assert cfg.scheduler._target_.split('.')[-1] == 'ReduceLROnPlateau'
    assert cfg.scheduler.mode == 'min'
    assert cfg.scheduler.factor == 0.1
    assert cfg.scheduler.patience == 10
    assert cfg.scheduler.threshold == 0.0001
    assert cfg.scheduler.threshold_mode == 'rel'
    assert cfg.scheduler.cooldown == 0
    assert cfg.scheduler.min_lr == 0
    assert cfg.scheduler.eps == 1e-08
    assert not cfg.scheduler.verbose
