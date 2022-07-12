from torch_geometric.testing import withPackage


@withPackage('hydra')
def test_config_store():
    import hydra
    from omegaconf import DictConfig

    from torch_geometric.graphgym.config_store import fill_config_store

    fill_config_store()

    with hydra.initialize(config_path='.'):
        cfg = hydra.compose(config_name='my_config')

    assert len(cfg) == 4
    assert 'dataset' in cfg
    assert 'model' in cfg
    assert 'optimizer' in cfg
    assert 'lr_scheduler' in cfg

    # Check `cfg.dataset`:
    assert len(cfg.dataset) == 2
    assert cfg.dataset._target_.split('.')[-1] == 'KarateClub'

    # Check `cfg.dataset.transform`:
    assert isinstance(cfg.dataset.transform, DictConfig)
    assert len(cfg.dataset.transform) == 2
    assert 'NormalizeFeatures' in cfg.dataset.transform
    assert 'AddSelfLoops' in cfg.dataset.transform

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
    assert len(cfg.model) == 12
    assert cfg.model._target_.split('.')[-1] == 'GCN'
    assert cfg.model.in_channels == 34
    assert cfg.model.out_channels == 4
    assert cfg.model.hidden_channels == 16
    assert cfg.model.num_layers == 2
    assert cfg.model.dropout == 0.0
    assert cfg.model.act == 'relu'
    assert cfg.model.norm is None
    assert cfg.model.norm_kwargs is None
    assert cfg.model.jk is None
    assert not cfg.model.act_first
    assert cfg.model.act_kwargs is None

    # Check `cfg.optimizer`:
    assert len(cfg.optimizer) in {6, 7}  # Arguments changed across 1.10/1.11
    assert cfg.optimizer._target_.split('.')[-1] == 'Adam'
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.betas == [0.9, 0.999]
    assert cfg.optimizer.eps == 1e-08
    assert cfg.optimizer.weight_decay == 0
    assert not cfg.optimizer.amsgrad
    if hasattr(cfg.optimizer, 'maximize'):
        assert not cfg.optimizer.maximize

    # Check `cfg.lr_scheduler`:
    assert len(cfg.lr_scheduler) == 10
    assert cfg.lr_scheduler._target_.split('.')[-1] == 'ReduceLROnPlateau'
    assert cfg.lr_scheduler.mode == 'min'
    assert cfg.lr_scheduler.factor == 0.1
    assert cfg.lr_scheduler.patience == 10
    assert cfg.lr_scheduler.threshold == 0.0001
    assert cfg.lr_scheduler.threshold_mode == 'rel'
    assert cfg.lr_scheduler.cooldown == 0
    assert cfg.lr_scheduler.min_lr == 0
    assert cfg.lr_scheduler.eps == 1e-08
    assert not cfg.lr_scheduler.verbose
