"""
ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor
--------------------------------------------
Graph Neural Network (GNN) training suffers from low scalability on multi-core CPUs. 
Specificially, the performance often caps at 16 cores, and no improvement is observed when applying more than 16 cores.
ARGO is a runtime system that offers scalable performance by overlapping the computation and communication during GNN training.
With ARGO enabled, we are able to scale over 64 cores, allowing ARGO to speedup GNN training (in terms of epoch time) by up to 5.06x and 4.54x on a Xeon 8380H and a Xeon 6430L, respectively.
--------------------------------------------
Paper Link: https://arxiv.org/abs/2402.03671
"""

import time
from typing import Callable, Tuple

import dgl.multiprocessing as dmp
import psutil
from skopt import gp_minimize


class ARGO:

    def __init__(
        self,
        n_search=10,
        epoch=200,
        batch_size=4096,
        space=[(2, 8), (1, 4), (1, 32)],
        random_state=1,
    ):
        self.n_search = (
            n_search  # number of configuration searches the auto-tuner will conduct
        )
        self.epoch = epoch  # number of epochs of training
        self.batch_size = batch_size  # size of the mini-batch
        self.space = space  # range of the search space; [number of processes, number of samplers, number of trainers]
        self.random_state = (
            random_state  # number of random initializations before searching
        )
        self.acq_func = "EI"
        self.counter = [0]

    def core_binder(
        self, num_cpu_proc: int, n_samp: int, n_train: int, rank: int
    ) -> Tuple[
        list[int], list[int]
    ]:  # Core Binder: bind CPU cores to perform sampling and model propagation
        load_core, comp_core = [], []
        n = psutil.cpu_count(logical=False)
        size = num_cpu_proc
        num_of_samplers = n_samp
        load_core = list(range(n // size * rank, n // size * rank + num_of_samplers))
        comp_core = list(
            range(
                n // size * rank + num_of_samplers,
                n // size * rank + num_of_samplers + n_train,
            )
        )
        return load_core, comp_core

    def auto_tuning(
        self, train: Callable, args
    ) -> list[
        int
    ]:  # auto-tuner: runs Bayesian optimization to search for the optimal configuration (number of processes, samplers, trainers)
        ep = 1
        result = gp_minimize(
            lambda x: self.mp_engine(x, train, args, ep),
            dimensions=self.space,
            n_calls=self.n_search,
            random_state=self.random_state,
            acq_func=self.acq_func,
        )
        return result

    def mp_engine(
        self, x: list[int], train: Callable, args, ep: int
    ) -> (
        float
    ):  # Multi-Process Engine: launches multiple GNN training processes to overlap computation with communication
        # x: optimal configurations provided by the auto-tuner; x = [number of processes, samplers, trainers]
        # train: the GNN training function
        n_proc = x[0]
        n_samp = x[1]
        n_train = x[2]
        n_total = psutil.cpu_count(logical=False)

        if n_proc * (n_samp + n_train) > n_total:  # handling corner cases
            n_proc = 2
            n_samp = 2
            n_train = (n_total // n_proc) - n_samp

        processes = []
        cnt = self.counter
        b_size = (
            self.batch_size // n_proc
        )  # adjust batch size based on number of processes

        tik = time.time()
        for i in range(n_proc):
            load_core, comp_core = self.core_binder(n_proc, n_samp, n_train, i)
            p = dmp.Process(
                target=train,
                args=(*args, i, n_proc, comp_core, load_core, cnt, b_size, ep),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        t = time.time() - tik

        self.counter[0] = self.counter[0] + 1

        return t

    def run(self, train, args):  # The "run" function launches ARGO to traing GNN model
        result = self.auto_tuning(train, args)
        x = result.x  # obtained the optimal configuration
        self.mp_engine(
            x, train, args, ep=(self.epoch - self.n_search)
        )  # continue training with the optimal configurations
