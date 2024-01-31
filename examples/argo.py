import time

import dgl.multiprocessing as dmp
import psutil
from skopt import gp_minimize


class ARGO:
    def __init__(self, n_search=10, epoch=200, batch_size=4096,
                 space=[(2, 8), (1, 4), (1, 32)], random_state=1):
        self.n_search = n_search
        self.epoch = epoch
        self.batch_size = batch_size
        self.space = space
        self.random_state = random_state
        self.acq_func = 'EI'
        self.counter = [0]

    def core_binder(self, num_cpu_proc, n_samp, n_train, rank):
        load_core, comp_core = [], []
        n = psutil.cpu_count(logical=False)
        size = num_cpu_proc
        num_of_samplers = n_samp
        load_core = list(
            range(n // size * rank, n // size * rank + num_of_samplers))
        comp_core = list(
            range(n // size * rank + num_of_samplers,
                  n // size * rank + num_of_samplers + n_train))
        return load_core, comp_core

    def auto_tuning(self, train, args):
        ep = 1
        result = gp_minimize(lambda x: self.mp_engine(x, train, args, ep),
                             dimensions=self.space, n_calls=self.n_search,
                             random_state=self.random_state,
                             acq_func=self.acq_func)
        return result

    def mp_engine(self, x, train, args, ep):  # Multi-Process Engine
        #configurations provided by the auto-tuner
        n_proc = x[0]
        n_samp = x[1]
        n_train = x[2]
        n_total = psutil.cpu_count(logical=False)

        if n_proc * (n_samp + n_train) > n_total:  #handling corner cases
            n_proc = 2
            n_samp = 2
            n_train = (n_total // n_proc) - n_samp

        processes = []
        cnt = self.counter
        b_size = (self.batch_size // n_proc)  # adjust batch size

        tik = time.time()
        for i in range(n_proc):
            load_core, comp_core = self.core_binder(n_proc, n_samp, n_train, i)
            p = dmp.Process(
                target=train,
                args=(*args, i, n_proc, comp_core, load_core, cnt, b_size, ep))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        t = time.time() - tik

        self.counter[0] = self.counter[0] + 1

        return t

    # args = arguments dataset
    def run(self, train, args):
        #auto-tuning
        result = self.auto_tuning(train, args)
        x = result.x  #optimal configuration
        self.mp_engine(x, train, args, ep=(self.epoch - self.n_search))
