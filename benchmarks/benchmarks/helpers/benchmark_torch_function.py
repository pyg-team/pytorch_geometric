from torch.utils.benchmark import Timer


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
    )
    return t0.blocked_autorange().median * 1e6


def benchmark_torch_function_in_milliseconds(f, *args, **kwargs):
    t0 = Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
    )
    return t0.blocked_autorange().median * 1e3
