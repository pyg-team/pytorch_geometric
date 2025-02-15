import os.path as osp
import warnings
from collections import namedtuple

import pytest
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym import register
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import LoggerCallback, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNStackStage
from torch_geometric.graphgym.models.head import GNNNodeHead
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils import (
    agg_runs,
    auto_select_device,
    params_count,
)
from torch_geometric.testing import onlyLinux, onlyOnline, withPackage

num_trivial_metric_calls = 0

Args = namedtuple('Args', ['cfg_file', 'opts'])
root = osp.join(osp.dirname(osp.realpath(__file__)))
args = Args(osp.join(root, 'example_node.yml'), [])


def trivial_metric(true, pred, task_type):
    global num_trivial_metric_calls
    num_trivial_metric_calls += 1
    return 1


@onlyOnline
@withPackage('yacs', 'pytorch_lightning')
@pytest.mark.parametrize('auto_resume', [True, False])
@pytest.mark.parametrize('skip_train_eval', [True, False])
@pytest.mark.parametrize('use_trivial_metric', [True, False])
def test_run_single_graphgym(tmp_path, capfd, auto_resume, skip_train_eval,
                             use_trivial_metric):
    warnings.filterwarnings('ignore', ".*does not have many workers.*")
    warnings.filterwarnings('ignore', ".*lower value for log_every_n_steps.*")

    load_cfg(cfg, args)
    cfg.out_dir = osp.join(tmp_path, 'out_dir')
    cfg.run_dir = osp.join(tmp_path, 'run_dir')
    cfg.dataset.dir = osp.join('/', 'tmp', 'pyg_test_datasets', 'Planetoid')
    cfg.train.auto_resume = auto_resume

    set_out_dir(cfg.out_dir, args.cfg_file)
    dump_cfg(cfg)
    set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    set_run_dir(cfg.out_dir)

    cfg.train.skip_train_eval = skip_train_eval
    cfg.train.enable_ckpt = use_trivial_metric and skip_train_eval
    if use_trivial_metric:
        if 'trivial' not in register.metric_dict:
            register.register_metric('trivial', trivial_metric)
        global num_trivial_metric_calls
        num_trivial_metric_calls = 0
        cfg.metric_best = 'trivial'
        cfg.custom_metrics = ['trivial']
    else:
        cfg.metric_best = 'auto'
        cfg.custom_metrics = []

    datamodule = GraphGymDataModule()
    assert len(datamodule.loaders) == 3

    model = create_model()
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.encoder, FeatureEncoder)
    assert isinstance(model.mp, GNNStackStage)
    assert isinstance(model.post_mp, GNNNodeHead)
    assert len(list(model.pre_mp.children())) == cfg.gnn.layers_pre_mp

    optimizer, scheduler = model.configure_optimizers()
    assert isinstance(optimizer[0], torch.optim.Adam)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.CosineAnnealingLR)

    cfg.params = params_count(model)
    assert cfg.params == 23883

    train(model, datamodule, logger=True,
          trainer_config={"enable_progress_bar": False})

    assert osp.isdir(get_ckpt_dir()) is cfg.train.enable_ckpt

    agg_runs(cfg.out_dir, cfg.metric_best)

    out, _ = capfd.readouterr()
    assert "train: {'epoch': 0," in out
    assert "val: {'epoch': 0," in out
    assert "train: {'epoch': 5," in out
    assert "val: {'epoch': 5," in out


@onlyOnline
@withPackage('yacs', 'pytorch_lightning')
def test_graphgym_module(tmp_path):
    import pytorch_lightning as pl

    load_cfg(cfg, args)
    cfg.out_dir = osp.join(tmp_path, 'out_dir')
    cfg.run_dir = osp.join(tmp_path, 'run_dir')
    cfg.dataset.dir = osp.join('/', 'tmp', 'pyg_test_datasets', 'Planetoid')

    set_out_dir(cfg.out_dir, args.cfg_file)
    dump_cfg(cfg)
    set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    set_run_dir(cfg.out_dir)

    loaders = create_loader()
    assert len(loaders) == 3

    model = create_model()
    assert isinstance(model, pl.LightningModule)

    optimizer, scheduler = model.configure_optimizers()
    assert isinstance(optimizer[0], torch.optim.Adam)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.CosineAnnealingLR)

    cfg.params = params_count(model)
    assert cfg.params == 23883

    keys = {"loss", "true", "pred_score", "step_end_time"}
    # test training step
    batch = next(iter(loaders[0]))
    batch.to(model.device)
    outputs = model.training_step(batch)
    assert keys == set(outputs.keys())
    assert isinstance(outputs["loss"], torch.Tensor)

    # test validation step
    batch = next(iter(loaders[1]))
    batch.to(model.device)
    outputs = model.validation_step(batch)
    assert keys == set(outputs.keys())
    assert isinstance(outputs["loss"], torch.Tensor)

    # test test step
    batch = next(iter(loaders[2]))
    batch.to(model.device)
    outputs = model.test_step(batch)
    assert keys == set(outputs.keys())
    assert isinstance(outputs["loss"], torch.Tensor)


@pytest.fixture
def destroy_process_group():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@onlyOnline
@onlyLinux
@withPackage('yacs', 'pytorch_lightning')
def test_train(destroy_process_group, tmp_path, capfd):
    warnings.filterwarnings('ignore', ".*does not have many workers.*")

    import pytorch_lightning as pl

    load_cfg(cfg, args)
    cfg.out_dir = osp.join(tmp_path, 'out_dir')
    cfg.run_dir = osp.join(tmp_path, 'run_dir')
    cfg.dataset.dir = osp.join('/', 'tmp', 'pyg_test_datasets', 'Planetoid')

    set_out_dir(cfg.out_dir, args.cfg_file)
    dump_cfg(cfg)
    set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    set_run_dir(cfg.out_dir)

    loaders = create_loader()
    model = create_model()
    cfg.params = params_count(model)
    logger = LoggerCallback()
    trainer = pl.Trainer(max_epochs=1, max_steps=4, callbacks=logger,
                         log_every_n_steps=1)
    train_loader, val_loader = loaders[0], loaders[1]
    trainer.fit(model, train_loader, val_loader)

    out, err = capfd.readouterr()
    assert 'Sanity Checking' in out
    assert 'Epoch 0:' in out
