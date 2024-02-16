"""ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor
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
import numpy as np
import psutil
from skopt import gp_minimize
from skopt.space import Normalize


def transform(self, X):
    X = np.asarray(X)
    if self.is_int:
        if np.any(np.round(X) > self.high):
            raise ValueError("All integer values should"
                             "be less than %f" % self.high)
        if np.any(np.round(X) < self.low):
            raise ValueError("All integer values should"
                             "be greater than %f" % self.low)
    else:
        if np.any(X > self.high + self._eps):
            raise ValueError("All values should"
                             "be less than %f" % self.high)
        if np.any(X < self.low - self._eps):
            raise ValueError("All values should"
                             "be greater than %f" % self.low)
    if (self.high - self.low) == 0.0:
        return X * 0.0
    if self.is_int:
        return (np.round(X).astype(int) - self.low) / (self.high - self.low)
    else:
        return (X - self.low) / (self.high - self.low)


def inverse_transform(self, X):
    X = np.asarray(X)
    if np.any(X > 1.0 + self._eps):
        raise ValueError("All values should be less than 1.0")
    if np.any(X < 0.0 - self._eps):
        raise ValueError("All values should be greater than 0.0")
    X_orig = X * (self.high - self.low) + self.low
    if self.is_int:
        return np.round(X_orig).astype(int)
    return X_orig


Normalize.transform = transform
Normalize.inverse_transform = inverse_transform


class ARGO:
    def __init__(
        self,
        n_search=10,
        epoch=200,
        batch_size=4096,
        space=[(2, 8), (1, 4), (1, 32)],
        random_state=1,
    ):
        """Initialize ARGO.

        Args:
            n_search: int
                Number of configuration searches the auto-tuner will conduct

            epoch: int
                Number of epochs of GNN training

            batch_size: int
                Size of the mini-batch

            space: list[Tuple(int,int)]
                Range of the search space; [range of processes, range of samplers for each process, range of trainers for each process]

            random_state: int
                Number of random initializations before searching

        """
        self.n_search = n_search
        self.epoch = epoch
        self.batch_size = batch_size
        self.space = space
        self.random_state = random_state
        self.acq_func = "EI"
        self.counter = [0]

    def core_binder(self, num_cpu_proc: int, n_samp: int, n_train: int,
                    rank: int) -> Tuple[list[int], list[int]]:
        """Produces lists of CPUs for each GNN training process for core binding.

        The Core Binder binds CPU cores to perform sampling (i.e., sampling cores) and model propagation (i.e., training cores).
        The actual binding is done using the CPU affinity function in the data_loader.
        The core_binder function here is used to produce the list of CPU IDs for the CPU affinity function.

        Args:
            num_cpu_proc: int
                Number of processes instantiated

            n_samp: int
                Number of sampling cores for each process

            n_train: int
                Number of training cores for each process

            rank: int
                The rank of the current process

        Returns: Tuple[list[int], list[int]]
            load_core: list[int]
                For a given process rank, the load_core specifies a list of CPU core IDs to be used for sampling, the length of load_core = n_samp.

            comp_core: list[int]
                For a given process rank, the comp_core specifies a list of CPU core IDs to be used for training, the length of comp_core = n_comp.

            .. note:: each process is assigned with a unique list of sampling cores and training cores, and no CPU core will appear in two lists or more.

        """
        load_core, comp_core = [], []
        n = psutil.cpu_count(logical=False)
        size = num_cpu_proc
        num_of_samplers = n_samp
        load_core = list(
            range(n // size * rank, n // size * rank + num_of_samplers))
        comp_core = list(
            range(
                n // size * rank + num_of_samplers,
                n // size * rank + num_of_samplers + n_train,
            ))
        return load_core, comp_core

    def auto_tuning(self, train: Callable, args) -> list[int]:
        """For a given training function, auto-tuner searches for the optimal configuration that leads to the shortest epoch time.

        The auto-tuner runs Bayesian Optimization (BO) to search for the optimal configuration (number of processes, samplers, trainers).
        During the search, the auto-tuner explores the design space by collecting the epoch time of various configurations.
        Specifically, the exploration is done by feeding the Multi-Process Engine with various configurations, and record the epoch time.
        After the searching is done, the optimal configuration will be used repeatedly until the end of model training.

        Args:
            train: Callable
                The GNN training function.

        Args:
                The inputs of the GNN training function.

        Returns:
            result: list[int]
                The optimal configurations (which leads to the shortest epoch time) found by running BO.
                - result[0]: number of processes to instantiate
                - result[1]: number of sampling cores for each process
                - result[2]: number of training cores for each process

        """
        ep = 1
        result = gp_minimize(
            lambda x: self.mp_engine(x, train, args, ep),
            dimensions=self.space,
            n_calls=self.n_search,
            random_state=self.random_state,
            acq_func=self.acq_func,
        )
        return result

    def mp_engine(self, x: list[int], train: Callable, args, ep: int) -> float:
        """Multi-Process Engine (MP Engine) launches multiple GNN training processes to overlap computation with communication.

        The MP Engine launches multiple GNN training processes in parallel to overlap computation with communication.
        Such an approach effectively improves the utilization of the memory bandwidth and the CPU cores.
        The MP Engine also adjust the batch size according to the number of processes instantiated, so that the effective batch size remains the same as the original program without ARGO.

        Args:
            x: list[int]
                Optimal configurations provided by the auto-tuner.
                - x[0]: number of processes to instantiate
                - x[1]: number of sampling cores for each process
                - x[2]: number of training cores for each process

            train: Callable
                The GNN training function.

        Args:
                The inputs of the GNN training function.

            ep: int
                number of epochs.

        Returns:
            t: float
                The epoch time using the current configuration `x`.
        """
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
        b_size = (self.batch_size // n_proc
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

    def run(self, train, args):
        """The `run` function launches ARGO to traing GNN model.

        The `run` function consists of three steps:
            - Step 1: run the auto-tuner to search for the optimal configuration
            - Step 2: record the optimal configuration
            - Step 3: use the optimal configuration repeatedly until the end of the model training

        Args:
            train: Callable
                The GNN training function.

        Args:
                The inputs of the GNN training function.

        """
        result = self.auto_tuning(train, args)  # Step 1
        x = result.x  # Step 2
        self.mp_engine(x, train, args,
                       ep=(self.epoch - self.n_search))  # Step 3
