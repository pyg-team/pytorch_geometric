import os
import random
import numpy as np
import torch
import logging

from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg, update_out_dir,
                             get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.contrib.train import *
from graphgym.register import train_dict

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        cfg.seed = i + 1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        update_out_dir(out_dir_parent, args.cfg_file)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()
        # Set learning environment
        loaders = create_loader()
        meters = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](meters, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(get_parent_dir(out_dir_parent, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
