import functools
import inspect
import logging
import os
import os.path as osp
import warnings
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import torch_geometric.graphgym.register as register
from torch_geometric.io import fs

try:  # Define global config object
    from yacs.config import CfgNode as CN
    cfg = CN()
except ImportError:
    cfg = None
    warnings.warn(
        "Could not define global config object. Please install "
        "'yacs' via 'pip install yacs' in order to use GraphGym", stacklevel=2)


def set_cfg(cfg):
    r"""This function sets the default config value.

    1) Note that for an experiment, only part of the arguments will be used
       The remaining unused arguments won't affect anything.
       So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, *e.g.*,
       :obj:`cfg.dataset.name`.

    :return: Configuration use by the experiment.
    """
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #

    # Set print destination: stdout / file / both
    cfg.print = 'both'

    # Select device: 'cpu', 'cuda', 'auto'
    cfg.accelerator = 'auto'

    # number of devices: eg. for 2 GPU set cfg.devices=2
    cfg.devices = 1

    # Output directory
    cfg.out_dir = 'results'

    # Config name (in out_dir)
    cfg.cfg_dest = 'config.yaml'

    # Names of registered custom metric funcs to be used (use defaults if none)
    cfg.custom_metrics = []

    # Random seed
    cfg.seed = 0

    # Print rounding
    cfg.round = 4

    # Tensorboard support for each run
    cfg.tensorboard_each_run = False

    # Tensorboard support for aggregated results
    cfg.tensorboard_agg = True

    # Additional num of worker for data loading
    cfg.num_workers = 0

    # Max threads used by PyTorch
    cfg.num_threads = 6

    # The metric for selecting the best epoch for each run
    cfg.metric_best = 'auto'

    # argmax or argmin in aggregating results
    cfg.metric_agg = 'argmax'

    # If visualize embedding.
    cfg.view_emb = False

    # If get GPU usage
    cfg.gpu_mem = False

    # If do benchmark analysis
    cfg.benchmark = False

    # ----------------------------------------------------------------------- #
    # Globally shared variables:
    # These variables will be set dynamically based on the input dataset
    # Do not directly set them here or in .yaml files
    # ----------------------------------------------------------------------- #

    cfg.share = CN()

    # Size of input dimension
    cfg.share.dim_in = 1

    # Size of out dimension, i.e., number of labels to be predicted
    cfg.share.dim_out = 1

    # Number of dataset splits: train/val/test
    cfg.share.num_splits = 1

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = 'Cora'

    # if PyG: look for it in Pytorch Geometric dataset
    # if NetworkX/nx: load data in NetworkX format
    cfg.dataset.format = 'PyG'

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = './datasets'

    # Task: node, edge, graph, link_pred
    cfg.dataset.task = 'node'

    # Type of task: classification, regression, classification_binary
    # classification_multi
    cfg.dataset.task_type = 'classification'

    # Transductive / Inductive
    # Graph classification is always inductive
    cfg.dataset.transductive = True

    # Split ratio of dataset. Len=2: Train, Val. Len=3: Train, Val, Test
    cfg.dataset.split = [0.8, 0.1, 0.1]

    # Whether to shuffle the graphs for splitting
    cfg.dataset.shuffle_split = True

    # Whether random split or use custom split: random / custom
    cfg.dataset.split_mode = 'random'

    # Whether to use an encoder for general attribute features
    cfg.dataset.encoder = True

    # Name of general encoder
    cfg.dataset.encoder_name = 'db'

    # If add batchnorm after general encoder
    cfg.dataset.encoder_bn = True

    # Whether to use an encoder for the node features
    cfg.dataset.node_encoder = False

    # Name of node encoder
    cfg.dataset.node_encoder_name = 'Atom'

    # If add batchnorm after node encoder
    cfg.dataset.node_encoder_bn = True

    # Whether to use an encoder for the edge features
    cfg.dataset.edge_encoder = False

    # Name of edge encoder
    cfg.dataset.edge_encoder_name = 'Bond'

    # If add batchnorm after edge encoder
    cfg.dataset.edge_encoder_bn = True

    # Dimension of the encoded features.
    # For now the node and edge encoding dimensions
    # are the same.
    cfg.dataset.encoder_dim = 128

    # Dimension for edge feature. Updated by the real dim of the dataset
    cfg.dataset.edge_dim = 128

    # ============== Link/edge tasks only

    # all or disjoint
    cfg.dataset.edge_train_mode = 'all'

    # Used in disjoint edge_train_mode. The proportion of edges used for
    # message-passing
    cfg.dataset.edge_message_ratio = 0.8

    # The ratio of negative samples to positive samples
    cfg.dataset.edge_negative_sampling_ratio = 1.0

    # Whether resample disjoint when dataset.edge_train_mode is 'disjoint'
    cfg.dataset.resample_disjoint = False

    # Whether resample negative edges at training time (link prediction only)
    cfg.dataset.resample_negative = False

    # What transformation function is applied to the dataset
    cfg.dataset.transform = 'none'

    # Whether cache the splitted dataset
    # NOTE: it should be cautiouslly used, as cached dataset may not have
    # exactly the same setting as the config file
    cfg.dataset.cache_save = False
    cfg.dataset.cache_load = False

    # Whether remove the original node features in the dataset
    cfg.dataset.remove_feature = False

    # Simplify TU dataset for synthetic tasks
    cfg.dataset.tu_simple = True

    # Convert to undirected graph (save 2*E edges)
    cfg.dataset.to_undirected = False

    # dataset location: local, snowflake
    cfg.dataset.location = 'local'

    # Define label: Table name
    cfg.dataset.label_table = 'none'

    # Define label: Column name
    cfg.dataset.label_column = 'none'

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CN()

    # Total graph mini-batch size
    cfg.train.batch_size = 16

    # Sampling strategy for a train loader
    cfg.train.sampler = 'full_batch'

    # Minibatch node
    cfg.train.sample_node = False

    # Num of sampled node per graph
    cfg.train.node_per_graph = 32

    # Radius: same, extend. same: same as cfg.gnn.layers_mp, extend: layers+1
    cfg.train.radius = 'extend'

    # Evaluate model on test data every eval period epochs
    cfg.train.eval_period = 10

    # Option to skip training epoch evaluation
    cfg.train.skip_train_eval = False

    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 100

    # Enabling checkpoint, set False to disable and save I/O
    cfg.train.enable_ckpt = True

    # Resume training from the latest checkpoint in the output directory
    cfg.train.auto_resume = False

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True

    # Number of iterations per epoch (for sampling based loaders only)
    cfg.train.iter_per_epoch = 32

    # GraphSAINTRandomWalkSampler: random walk length
    cfg.train.walk_length = 4

    # NeighborSampler: number of sampled nodes per layer
    cfg.train.neighbor_sizes = [20, 15, 10, 5]

    # ----------------------------------------------------------------------- #
    # Validation options
    # ----------------------------------------------------------------------- #
    cfg.val = CN()

    # Minibatch node
    cfg.val.sample_node = False

    # Sampling strategy for a val/test loader
    cfg.val.sampler = 'full_batch'

    # Num of sampled node per graph
    cfg.val.node_per_graph = 32

    # Radius: same, extend. same: same as cfg.gnn.layers_mp, extend: layers+1
    cfg.val.radius = 'extend'

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN()

    # Model type to use
    cfg.model.type = 'gnn'

    # Auto match computational budget, match upper bound / lower bound
    cfg.model.match_upper = True

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'cross_entropy'

    # size average for loss function. 'mean' or 'sum'
    cfg.model.size_average = 'mean'

    # Threshold for binary classification
    cfg.model.thresh = 0.5

    # ============== Link/edge tasks only
    # Edge decoding methods.
    #   - dot: compute dot(u, v) to predict link (binary)
    #   - cosine_similarity: use cosine similarity (u, v) to predict link (
    #   binary)
    #   - concat: use u||v followed by an nn.Linear to obtain edge embedding
    #   (multi-class)
    cfg.model.edge_decoding = 'dot'
    # ===================================

    # ================== Graph tasks only
    # Pooling methods.
    #   - add: global add pool
    #   - mean: global mean pool
    #   - max: global max pool
    cfg.model.graph_pooling = 'add'
    # ===================================

    # ----------------------------------------------------------------------- #
    # GNN options
    # ----------------------------------------------------------------------- #
    cfg.gnn = CN()

    # Prediction head. Use cfg.dataset.task by default
    cfg.gnn.head = 'default'

    # Number of layers before message passing
    cfg.gnn.layers_pre_mp = 0

    # Number of layers for message passing
    cfg.gnn.layers_mp = 2

    # Number of layers after message passing
    cfg.gnn.layers_post_mp = 0

    # Hidden layer dim. Automatically set if train.auto_match = True
    cfg.gnn.dim_inner = 16

    # Type of graph conv: generalconv, gcnconv, sageconv, gatconv, ...
    cfg.gnn.layer_type = 'generalconv'

    # Stage type: 'stack', 'skipsum', 'skipconcat'
    cfg.gnn.stage_type = 'stack'

    # How many layers to skip each time
    cfg.gnn.skip_every = 1

    # Whether use batch norm
    cfg.gnn.batchnorm = True

    # Activation
    cfg.gnn.act = 'relu'

    # Dropout
    cfg.gnn.dropout = 0.0

    # Aggregation type: add, mean, max
    # Note: only for certain layers that explicitly set aggregation type
    # e.g., when cfg.gnn.layer_type = 'generalconv'
    cfg.gnn.agg = 'add'

    # Normalize adj
    cfg.gnn.normalize_adj = False

    # Message direction: single, both
    cfg.gnn.msg_direction = 'single'

    # Whether add message from node itself: none, add, cat
    cfg.gnn.self_msg = 'concat'

    # Number of attention heads
    cfg.gnn.att_heads = 1

    # After concat attention heads, add a linear layer
    cfg.gnn.att_final_linear = False

    # After concat attention heads, add a linear layer
    cfg.gnn.att_final_linear_bn = False

    # Normalize after message passing
    cfg.gnn.l2norm = True

    # randomly use fewer edges for message passing
    cfg.gnn.keep_edge = 0.5

    # clear cached feature_new
    cfg.gnn.clear_feature = True

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 200

    # ----------------------------------------------------------------------- #
    # Batch norm options
    # ----------------------------------------------------------------------- #
    cfg.bn = CN()

    # BN epsilon
    cfg.bn.eps = 1e-5

    # BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
    cfg.bn.mom = 0.1

    # ----------------------------------------------------------------------- #
    # Memory options
    # ----------------------------------------------------------------------- #
    cfg.mem = CN()

    # Perform ReLU inplace
    cfg.mem.inplace = False

    # Set user customized cfgs
    for func in register.config_dict.values():
        func(cfg)


def assert_cfg(cfg):
    r"""Checks config values, do necessary post processing to the configs."""
    if cfg.dataset.task not in ['node', 'edge', 'graph', 'link_pred']:
        raise ValueError(f"Task '{cfg.dataset.task}' not supported. Must be "
                         f"one of node, edge, graph, link_pred")
    if 'classification' in cfg.dataset.task_type and cfg.model.loss_fun == \
            'mse':
        cfg.model.loss_fun = 'cross_entropy'
        logging.warning(
            'model.loss_fun changed to cross_entropy for classification.')
    if cfg.dataset.task_type == 'regression' and cfg.model.loss_fun == \
            'cross_entropy':
        cfg.model.loss_fun = 'mse'
        logging.warning('model.loss_fun changed to mse for regression.')
    if cfg.dataset.task == 'graph' and cfg.dataset.transductive:
        cfg.dataset.transductive = False
        logging.warning('dataset.transductive changed '
                        'to False for graph task.')
    if cfg.gnn.layers_post_mp < 1:
        cfg.gnn.layers_post_mp = 1
        logging.warning('Layers after message passing should be >=1')
    if cfg.gnn.head == 'default':
        cfg.gnn.head = cfg.dataset.task
    cfg.run_dir = cfg.out_dir


def dump_cfg(cfg):
    r"""Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg_file = osp.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_cfg(cfg, args):
    r"""Load configurations from file system and command line.

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser
    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)


def makedirs_rm_exist(dir):
    if osp.isdir(dir):
        fs.rm(dir)
    os.makedirs(dir, exist_ok=True)


def get_fname(fname):
    r"""Extract filename from file name path.

    Args:
        fname (str): Filename for the yaml format configuration file
    """
    fname = osp.basename(fname)
    if fname.endswith('.yaml'):
        fname = fname[:-5]
    elif fname.endswith('.yml'):
        fname = fname[:-4]
    return fname


def set_out_dir(out_dir, fname):
    r"""Create the directory for full experiment run.

    Args:
        out_dir (str): Directory for output, specified in :obj:`cfg.out_dir`
        fname (str): Filename for the yaml format configuration file
    """
    fname = get_fname(fname)
    cfg.out_dir = osp.join(out_dir, fname)
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)


def set_run_dir(out_dir):
    r"""Create the directory for each random seed experiment run.

    Args:
        out_dir (str): Directory for output, specified in :obj:`cfg.out_dir`
    """
    cfg.run_dir = osp.join(out_dir, str(cfg.seed))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


set_cfg(cfg)


def from_config(func):
    if inspect.isclass(func):
        params = list(inspect.signature(func.__init__).parameters.values())[1:]
    else:
        params = list(inspect.signature(func).parameters.values())

    arg_names = [p.name for p in params]
    has_defaults = [p.default != inspect.Parameter.empty for p in params]

    @functools.wraps(func)
    def wrapper(*args, cfg: Any = None, **kwargs):
        if cfg is not None:
            cfg = dict(cfg) if isinstance(cfg, Iterable) else asdict(cfg)

            iterator = zip(arg_names[len(args):], has_defaults[len(args):])
            for arg_name, has_default in iterator:
                if arg_name in kwargs:
                    continue
                elif arg_name in cfg:
                    kwargs[arg_name] = cfg[arg_name]
                elif not has_default:
                    raise ValueError(f"'cfg.{arg_name}' undefined")
        return func(*args, **kwargs)

    return wrapper
