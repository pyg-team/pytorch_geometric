import os.path as osp
import random
import shutil
import sys
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
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNStackStage
from torch_geometric.graphgym.models.head import GNNNodeHead
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils import (
    agg_runs,
    auto_select_device,
    params_count,
)
from torch_geometric.testing import withPackage

num_trivial_metric_calls = 0


def trivial_metric(true, pred, task_type):
    global num_trivial_metric_calls
    num_trivial_metric_calls += 1
    return 1


@withPackage('yacs')
@pytest.mark.parametrize('auto_resume', [True, False])
@pytest.mark.parametrize('skip_train_eval', [True, False])
@pytest.mark.parametrize('use_trivial_metric', [True, False])
def test_run_single_graphgym(auto_resume, skip_train_eval, use_trivial_metric):
    Args = namedtuple('Args', ['cfg_file', 'opts'])
    root = osp.join(osp.dirname(osp.realpath(__file__)))
    args = Args(osp.join(root, 'example_node.yml'), [])

    load_cfg(cfg, args)
    cfg.out_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    cfg.run_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
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

    loaders = create_loader()
    assert len(loaders) == 3

    loggers = create_logger()
    assert len(loggers) == 3

    model = create_model()
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.encoder, FeatureEncoder)
    assert isinstance(model.mp, GNNStackStage)
    assert isinstance(model.post_mp, GNNNodeHead)
    assert len(list(model.pre_mp.children())) == cfg.gnn.layers_pre_mp

    optimizer = create_optimizer(model.parameters(), cfg.optim)
    assert isinstance(optimizer, torch.optim.Adam)

    scheduler = create_scheduler(optimizer, cfg.optim)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    cfg.params = params_count(model)
    assert cfg.params == 23880

    train(loggers, loaders, model, optimizer, scheduler)

    if use_trivial_metric:
        # 6 total epochs, 4 eval epochs, 3 splits (1 training split)
        assert num_trivial_metric_calls == 12 if skip_train_eval else 14

    assert osp.isdir(get_ckpt_dir()) is cfg.train.enable_ckpt

    agg_runs(cfg.out_dir, cfg.metric_best)

    shutil.rmtree(cfg.out_dir)
