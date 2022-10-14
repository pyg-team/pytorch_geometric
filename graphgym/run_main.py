import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
# from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/ring_transfer.yaml'

# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/example.yaml'
# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/example_graph_copy.yaml'
# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/example_graph_2.yaml'
# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/configs_03.10/Cora_V_2MP.yaml'
# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/lrgb_peptides-func-GCN.yaml'

# REPRODUCIBILITY TEST
# argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/ring_transfer_REPRO_1.yaml'
argpath = '/Users/beng/Documents/pytorch_geometric/graphgym/configs/ring_transfer_REPRO_2.yaml'

repeat = 1

import argparse
def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args("--cfg {} --repeat {}".format(argpath, repeat).split())
    

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        datamodule = GraphGymDataModule()
        model = create_model()
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        train(model, datamodule, logger=True)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
