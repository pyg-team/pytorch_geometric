import gc
import time
import psutil
import os

import numpy as np
import torch


def get_peak_memory(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    elif device == "cpu":
        total = psutil.virtual_memory().total
        percentage = psutil.Process(os.getpid()).memory_percent()
        return percentage * total / 2**30

    raise NotImplementedError(f"Device {device} not supported")



def benchmark_train(
    model,
    device="cuda",
    batch_size=256,
    epochs=10,
    dataset="cifar10",
):
    """
    Arguments:
        device: Device to use for training
        batch_size: Batch size to use for training
        epochs: Number of epochs to train
        model: Model to use for training
        dataset: Dataset to use for training

    Returns:
        Train losses
        Validation losses
        Memory usage
        Median of time per epoch
        Ratio of initial memory to final memory  (memory leak)
    """

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()


    t0 = time.perf_counter()



    t1 = time.perf_counter()
    return {
        "time": t1 - t0,
        "peak_memory": get_peak_memory(device),
    }
