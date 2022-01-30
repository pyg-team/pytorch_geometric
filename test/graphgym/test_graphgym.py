import os.path as osp
import random
import shutil
import sys
from collections import namedtuple

import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_agg_dir, set_run_dir)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNStackStage
from torch_geometric.graphgym.models.head import GNNNodeHead
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils import (agg_runs, auto_select_device,
                                            params_count)


def test_run_single_graphgym():
    Args = namedtuple('Args', ['cfg_file', 'opts'])
    root = osp.join(osp.dirname(osp.realpath(__file__)))
    args = Args(osp.join(root, 'example_node.yml'), [])

    load_cfg(cfg, args)
    cfg.out_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    cfg.run_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    cfg.dataset.dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dump_cfg(cfg)
    set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    set_run_dir(cfg.out_dir, args.cfg_file)

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

    agg_runs(set_agg_dir(cfg.out_dir, args.cfg_file), cfg.metric_best)

    shutil.rmtree(cfg.out_dir)
    shutil.rmtree(cfg.dataset.dir)
