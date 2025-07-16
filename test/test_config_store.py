from typing import Any, Dict, List, Tuple

from torch_geometric.config_store import (
    class_from_dataclass,
    clear_config_store,
    dataclass_from_class,
    fill_config_store,
    get_config_store,
    map_annotation,
    register,
    to_dataclass,
)
from torch_geometric.testing import minPython, withPackage
from torch_geometric.transforms import AddSelfLoops


def teardown_function():
    clear_config_store()


def test_to_dataclass():
    from torch_geometric.transforms import AddSelfLoops

    AddSelfLoopsConfig = to_dataclass(AddSelfLoops, with_target=True)
    assert AddSelfLoopsConfig.__name__ == 'AddSelfLoops'

    fields = AddSelfLoopsConfig.__dataclass_fields__

    assert fields['attr'].name == 'attr'
    assert fields['attr'].type == str
    assert fields['attr'].default == 'edge_weight'

    assert fields['fill_value'].name == 'fill_value'
    assert fields['fill_value'].type == Any
    assert fields['fill_value'].default == 1.0

    assert fields['_target_'].name == '_target_'
    assert fields['_target_'].type == str
    assert fields['_target_'].default == (
        'torch_geometric.transforms.add_self_loops.AddSelfLoops')

    cfg = AddSelfLoopsConfig()
    assert str(cfg) == ("AddSelfLoops(attr='edge_weight', fill_value=1.0, "
                        "_target_='torch_geometric.transforms.add_self_loops."
                        "AddSelfLoops')")


@minPython('3.10')
def test_map_annotation():
    mapping = {int: Any}
    assert map_annotation(dict[str, int], mapping) == dict[str, Any]
    assert map_annotation(Dict[str, float], mapping) == Dict[str, float]
    assert map_annotation(List[str], mapping) == List[str]
    assert map_annotation(List[int], mapping) == List[Any]
    assert map_annotation(Tuple[int], mapping) == Tuple[Any]
    assert map_annotation(dict[str, int], mapping) == dict[str, Any]
    assert map_annotation(dict[str, float], mapping) == dict[str, float]
    assert map_annotation(list[str], mapping) == list[str]
    assert map_annotation(list[int], mapping) == list[Any]
    assert map_annotation(tuple[int], mapping) == tuple[Any]


def test_register():
    register(AddSelfLoops, group='transform')
    assert 'transform' in get_config_store().repo

    AddSelfLoopsConfig = dataclass_from_class('AddSelfLoops')

    Cls = class_from_dataclass('AddSelfLoops')
    assert Cls == AddSelfLoops
    Cls = class_from_dataclass(AddSelfLoopsConfig)
    assert Cls == AddSelfLoops

    ConfigCls = dataclass_from_class('AddSelfLoops')
    assert ConfigCls == AddSelfLoopsConfig
    ConfigCls = dataclass_from_class(ConfigCls)
    assert ConfigCls == AddSelfLoopsConfig


def test_fill_config_store():
    fill_config_store()

    assert {
        'transform',
        'dataset',
        'model',
        'optimizer',
        'lr_scheduler',
    }.issubset(get_config_store().repo.keys())


@withPackage('hydra')
def test_hydra_config_store():
    import hydra
    from omegaconf import DictConfig

    fill_config_store()

    with hydra.initialize(config_path='.', version_base='1.1'):
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
    assert cfg.dataset.transform.AddSelfLoops.fill_value == 1.0

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
    assert cfg.optimizer._target_.split('.')[-1] == 'Adam'
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.betas == [0.9, 0.999]
    assert cfg.optimizer.eps == 1e-08
    assert cfg.optimizer.weight_decay == 0
    assert not cfg.optimizer.amsgrad
    if hasattr(cfg.optimizer, 'maximize'):
        assert not cfg.optimizer.maximize

    # Check `cfg.lr_scheduler`:
    assert cfg.lr_scheduler._target_.split('.')[-1] == 'ReduceLROnPlateau'
    assert cfg.lr_scheduler.mode == 'min'
    assert cfg.lr_scheduler.factor == 0.1
    assert cfg.lr_scheduler.patience == 10
    assert cfg.lr_scheduler.threshold == 0.0001
    assert cfg.lr_scheduler.threshold_mode == 'rel'
    assert cfg.lr_scheduler.cooldown == 0
    assert cfg.lr_scheduler.min_lr == 0
    assert cfg.lr_scheduler.eps == 1e-08
