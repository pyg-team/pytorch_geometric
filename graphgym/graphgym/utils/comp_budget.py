import math
from graphgym.loader import create_dataset
from graphgym.model_builder import create_model
from graphgym.config import cfg, set_cfg
from yacs.config import CfgNode as CN

import pdb


def params_count(model):
    """Computes the number of parameters."""
    return sum([p.numel() for p in model.parameters()])


def get_stats():
    model = create_model(to_device=False, dim_in=1, dim_out=1)
    return params_count(model)


def match_computation(stats_baseline, key=['gnn', 'dim_inner'], mode='sqrt'):
    '''Match computation budge by cfg.gnn.dim_inner'''
    stats = get_stats()
    if stats != stats_baseline:
        # Phase 1: fast approximation
        while True:
            if mode == 'sqrt':
                scale = math.sqrt(stats_baseline / stats)
            elif mode == 'linear':
                scale = stats_baseline / stats
            step = int(round(cfg[key[0]][key[1]] * scale)) - cfg[key[0]][key[1]]
            cfg[key[0]][key[1]] += step
            stats = get_stats()
            if abs(step) <= 1:
                break
        # Phase 2: fine tune
        flag_init = 1 if stats < stats_baseline else -1
        step = 1
        while True:
            cfg[key[0]][key[1]] += flag_init * step
            stats = get_stats()
            flag = 1 if stats < stats_baseline else -1
            if stats == stats_baseline:
                return stats
            if flag != flag_init:
                if cfg.model.match_upper == False:  # stats is SMALLER
                    if flag < 0:
                        cfg[key[0]][key[1]] -= flag_init * step
                    return get_stats()
                else:
                    if flag > 0:
                        cfg[key[0]][key[1]] -= flag_init * step
                    return get_stats()
    return stats


def dict_to_stats(cfg_dict):
    set_cfg(cfg)
    cfg_new = CN(cfg_dict)
    cfg.merge_from_other_cfg(cfg_new)
    stats = get_stats()
    set_cfg(cfg)
    return stats


def dict_match_baseline(cfg_dict, cfg_dict_baseline, verbose=True):
    stats_baseline = dict_to_stats(cfg_dict_baseline)
    set_cfg(cfg)
    cfg_new = CN(cfg_dict)
    cfg.merge_from_other_cfg(cfg_new)
    stats = match_computation(stats_baseline, key=['gnn', 'dim_inner'])
    if 'gnn' in cfg_dict:
        cfg_dict['gnn']['dim_inner'] = cfg.gnn.dim_inner
    else:
        cfg_dict['gnn'] = {'dim_inner', cfg.gnn.dim_inner}
    set_cfg(cfg)
    if verbose:
        print('Computational budget has matched: Baseline params {}, '
              'Current params {}'.format(stats_baseline, stats))
    return cfg_dict

### test functionality
# stats_baseline = get_stats()
# match_computation(stats_baseline + 1000000)
