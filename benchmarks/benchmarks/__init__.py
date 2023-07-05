import os

RUN_ALL = bool(int(os.environ.get("PYG_BENCH_RUN_ALL", "0")))
RUN_CPU = RUN_ALL and bool(int(os.environ.get("PYG_BENCH_RUN_CPU", "0")))
RUN_CUDA = RUN_ALL and bool(int(os.environ.get("PYG_BENCH_RUN_CUDA", "1")))

devices = []

if RUN_CPU:
    devices.append("cpu")

if RUN_CUDA:
    devices.append("cuda")
