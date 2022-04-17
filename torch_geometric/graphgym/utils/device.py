import logging
import os
import subprocess

import numpy as np
import torch

from torch_geometric.graphgym.config import cfg


def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def get_current_gpu_usage():
    '''
    Get the current GPU memory usage.
    '''
    if cfg.gpu_mem and cfg.device != 'cpu' and torch.cuda.is_available():
        result = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        current_pid = os.getpid()
        used_memory = 0
        for line in result.strip().split('\n'):
            line = line.split(', ')
            if current_pid == int(line[0]):
                used_memory += int(line[1])
        return used_memory
    else:
        return -1


def auto_select_device(memory_max=8000, memory_bias=200, strategy='random'):
    r'''
    Auto select device for the experiment. Useful when having multiple GPUs.

    Args:
        memory_max (int): Threshold of existing GPU memory usage. GPUs with
        memory usage beyond this threshold will be deprioritized.
        memory_bias (int): A bias GPU memory usage added to all the GPUs.
        Avoild dvided by zero error.
        strategy (str, optional): 'random' (random select GPU) or 'greedy'
        (greedily select GPU)

    '''
    if cfg.device != 'cpu' and torch.cuda.is_available():
        if cfg.device == 'auto':
            memory_raw = get_gpu_memory_map()
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            elif strategy == 'random':
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                gpu_prob = memory / memory.sum()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.device = 'cuda:{}'.format(cuda)
    else:
        cfg.device = 'cpu'
