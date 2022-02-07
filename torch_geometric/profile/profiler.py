import functools
from collections import OrderedDict, defaultdict, namedtuple
from typing import List, NamedTuple, Tuple

import torch
import torch.profiler as torch_profiler

# predefined namedtuple for variable setting (global template)
Trace = namedtuple("Trace", ["path", "leaf", "module"])

# the metrics returned from the torch profiler
Measure = namedtuple("Measure", [
    "self_cpu_total", "cpu_total", "self_cuda_total", "cuda_total",
    "self_cpu_memory", "cpu_memory", "self_cuda_memory", "cuda_memory",
    "occurrences"
])


class Profiler(object):
    r"""Layer by layer profiling of PyTorch models, using the PyTorch profiler
    for memory profiling and part of the code is adapted from torchprof for
    layer-wise grouping (https://github.com/awwong1/torchprof).

    .. note::
        torch 1.8.1 and above is needed. For Windows OS, torch 1.9 is expected.

    Args:
        model (PyG model): the underlying model to be profiled. It should
            be a valid Pytorch nn model.
        enabled (bool, optional): If true, turn on the profiler.
            (default: :obj:`False`)
        use_cuda (bool, optional):
            (default: :obj:`False`)
        profile_memory (bool, optional): If True, also profile for the memory
            usage information.
            (default: :obj:`False`)
        paths (str, optional): Predefine path for fast loading. By default, it
            will not be used.
            (default: :obj:`False`)
    """
    def __init__(self, model, enabled=True, use_cuda=False,
                 profile_memory=False, paths=None):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("the profiler can be initialized only once")
        self.entered = True
        self._forwards = {}  # store the original forward functions

        # generate the trace and conduct profiling
        self.traces = tuple(map(self._hook_trace, _walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        full_table, heading, raw_info, layer_names, layer_stats = \
            layer_trace(self.traces,
                        self.trace_profile_events)
        # print layer-wise memory consumption
        print(full_table)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        """Add hooks to torch modules for profiling. The underlying model's
        forward pass is hooked/decorated here.
        """
        [path, leaf, module] = trace

        # the id of the model  is guaranteed to be unique
        _id = id(module)
        if (self.paths is not None
                and path in self.paths) or (self.paths is None and leaf):
            if _id in self._ids:
                # already wrapped
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                """The forward pass is decorated and profiled here
                """
                # only torch 1.8.1+ is supported
                torch_version = torch.__version__
                if torch_version <= '1.8.1':
                    raise NotImplementedError(
                        "Profiler requires at least torch 1.8.1")

                # profile the forward pass
                with torch_profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ], profile_memory=self.profile_memory) as prof:
                    res = _forward(*args, **kwargs)

                event_list = prof.events()

                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            # decorate the underlying model's forward pass
            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        """Clean it up after the profiling is done.
        """
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if (self.paths is not None
                and path in self.paths) or (self.paths is None and leaf):
            module.forward = self._forwards[path]


def layer_trace(
        traces: NamedTuple,
        trace_events: object,
        show_events: bool = True,
        paths: str = None,
        use_cuda: bool = False,
        profile_memory: bool = False,
        dt: Tuple[str] = ('-', '-', '-', ' '),
) -> object:
    """Construct human readable output of the profiler traces and events. The
    information is presented in layers, and each layer contains its underlying
    operators.

    Args:
        traces (trace object): Raw trace to be parsed.
        trace_events (trace object): Raw events to be parsed.
        show_events (bool, optional): If True, show detailed event information.
            (default: :obj:`True`)
        paths (str, optional): Predefine path for fast loading. By default, it
            will not be used.
            (default: :obj:`False`)
        use_cuda (bool, optional):
            (default: :obj:`False`)
        profile_memory (bool, optional): If True, also profile for the memory
            usage information.
            (default: :obj:`False`)
        dt (object, optional): Delimiters for showing the events.
    """
    tree = OrderedDict()

    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and ((paths is None and leaf) or
                                       (paths is not None and path in paths)):
                # tree measurements have key None, avoiding name conflict
                if show_events:
                    for event_name, event_group in group_by(
                            events, lambda e: e.name):
                        event_group = list(event_group)
                        current_tree[name][event_name] = {
                            None:
                            _build_measure_tuple(event_group, len(event_group))
                        }
                else:
                    current_tree[name][None] = _build_measure_tuple(
                        events, len(trace_events[path]))
            current_tree = current_tree[name]
    tree_lines = _flatten_tree(tree)

    format_lines = []
    has_self_cuda_total = False
    has_self_cpu_memory = False
    has_cpu_memory = False
    has_self_cuda_memory = False
    has_cuda_memory = False

    raw_results = {}
    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line

        next_depths = [pl[0] for pl in tree_lines[idx + 1:]]
        pre = "-"
        if depth > 0:
            pre = dt[1] if depth in next_depths and next_depths[
                0] >= depth else dt[2]
            depth -= 1
        while depth > 0:
            pre = (dt[0] + pre) if depth in next_depths else (dt[3] + pre)
            depth -= 1

        format_lines.append([pre + name, *_format_measure_tuple(measures)])
        if measures:
            has_self_cuda_total = (has_self_cuda_total
                                   or measures.self_cuda_total is not None)
            has_self_cpu_memory = (has_self_cpu_memory
                                   or measures.self_cpu_memory is not None)
            has_cpu_memory = has_cpu_memory or measures.cpu_memory is not None
            has_self_cuda_memory = (has_self_cuda_memory
                                    or measures.self_cuda_memory is not None)
            has_cuda_memory = (has_cuda_memory
                               or measures.cuda_memory is not None)

            raw_results[name] = [
                measures.self_cpu_total, measures.cpu_total,
                measures.self_cuda_total, measures.cuda_total,
                measures.self_cpu_memory, measures.cpu_memory,
                measures.self_cuda_memory, measures.cuda_memory,
                measures.occurrences
            ]

    # construct the table (this is pretty ugly and can probably be optimized)
    heading = (
        "Module",
        "Self CPU total",
        "CPU total",
        "Self CUDA total",
        "CUDA total",
        "Self CPU Mem",
        "CPU Mem",
        "Self CUDA Mem",
        "CUDA Mem",
        "Number of Calls",
    )

    # get the output aligned
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]

    # not all columns should be displayed, specify kept indexes
    keep_indexes = [0, 1, 2, 9]
    if profile_memory:
        if has_self_cpu_memory:
            keep_indexes.append(5)
        if has_cpu_memory:
            keep_indexes.append(6)
    if use_cuda:
        if has_self_cuda_total:
            keep_indexes.append(3)
        keep_indexes.append(4)
        if profile_memory:
            if has_self_cuda_memory:
                keep_indexes.append(7)
            if has_cuda_memory:
                keep_indexes.append(8)

    # the final columns to be shown
    keep_indexes = tuple(sorted(keep_indexes))

    heading_list = list(heading)

    display = (  # table heading
        " | ".join([
            "{:<{}s}".format(heading[keep_index], max_lens[keep_index])
            for keep_index in keep_indexes
        ]) + "\n")
    display += (  # separator
        "-|-".join([
            "-" * max_len for val_idx, max_len in enumerate(max_lens)
            if val_idx in keep_indexes
        ]) + "\n")
    for format_line in format_lines:  # body
        display += (" | ".join([
            "{:<{}s}".format(value, max_lens[val_idx])
            for val_idx, value in enumerate(format_line)
            if val_idx in keep_indexes
        ]) + "\n")
    # layer information readable
    key_dict = {}
    layer_names = []
    layer_stats = []
    for format_line in format_lines:  # body
        if format_line[1] == '':  # key line
            key_dict[format_line[0].count("-")] = format_line[0]
        else:  # must print
            # get current line's level
            curr_level = format_line[0].count("-")
            par_str = ""
            for i in range(1, curr_level):
                par_str += key_dict[i]
            curr_key = par_str + format_line[0]
            layer_names.append(curr_key)
            layer_stats.append(format_line[1:])

    return display, heading_list, raw_results, layer_names, layer_stats


def _flatten_tree(t, depth=0):
    """Internal method, not for external use.
    """
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(_flatten_tree(st, depth=depth + 1))
    return flat


def _build_measure_tuple(events: List, occurrences: List) -> NamedTuple:
    """This internal function builds the specific information for an event.
    """
    # memory profiling supported in torch >= 1.6
    self_cpu_memory = None
    has_self_cpu_memory = any(
        hasattr(e, "self_cpu_memory_usage") for e in events)
    if has_self_cpu_memory:
        self_cpu_memory = sum(
            [getattr(e, "self_cpu_memory_usage", 0) for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = sum([getattr(e, "cpu_memory_usage", 0) for e in events])
    self_cuda_memory = None
    has_self_cuda_memory = any(
        hasattr(e, "self_cuda_memory_usage") for e in events)
    if has_self_cuda_memory:
        self_cuda_memory = sum(
            [getattr(e, "self_cuda_memory_usage", 0) for e in events])
    cuda_memory = None
    has_cuda_memory = any(hasattr(e, "cuda_memory_usage") for e in events)
    if has_cuda_memory:
        cuda_memory = sum([getattr(e, "cuda_memory_usage", 0) for e in events])

    # self CUDA time supported in torch >= 1.7
    self_cuda_total = None
    has_self_cuda_time = any(
        hasattr(e, "self_cuda_time_total") for e in events)
    if has_self_cuda_time:
        self_cuda_total = sum(
            [getattr(e, "self_cuda_time_total", 0) for e in events])

    return Measure(
        self_cpu_total=sum([e.self_cpu_time_total for e in events]),
        cpu_total=sum([e.cpu_time_total for e in events]),
        self_cuda_total=self_cuda_total,
        cuda_total=sum([e.cuda_time_total for e in events]),
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def _format_measure_tuple(measure: NamedTuple) -> NamedTuple:
    """Internal method, not for external use.
    """
    self_cpu_total = (format_time(measure.self_cpu_total) if measure else "")
    cpu_total = format_time(measure.cpu_total) if measure else ""
    self_cuda_total = (format_time(measure.self_cuda_total) if measure
                       and measure.self_cuda_total is not None else "")
    cuda_total = format_time(measure.cuda_total) if measure else ""
    self_cpu_memory = (format_memory(measure.self_cpu_memory) if measure
                       and measure.self_cpu_memory is not None else "")
    cpu_memory = (format_memory(measure.cpu_memory)
                  if measure and measure.cpu_memory is not None else "")
    self_cuda_memory = (format_memory(measure.self_cuda_memory) if measure
                        and measure.self_cuda_memory is not None else "")
    cuda_memory = (format_memory(measure.cuda_memory)
                   if measure and measure.cuda_memory is not None else "")
    occurrences = str(measure.occurrences) if measure else ""

    return Measure(
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_cuda_total,
        cuda_total=cuda_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def group_by(events, keyfn):
    """Internal method, not for external use.
    """
    event_groups = OrderedDict()
    for event in events:
        key = keyfn(event)
        key_events = event_groups.get(key, [])
        key_events.append(event)
        event_groups[key] = key_events
    return event_groups.items()


def _walk_modules(module, name="", path=()):
    """Internal function only. Walk through a PyTorch Model and
    output Trace tuples (its path, whether has leaf, specific model)
    """

    # if not defined, use the class name of the model
    if not name:
        name = module.__class__.__name__

    # This will track the children of the module (layers)
    # for instance, [('conv1', GCNConv(10, 16)), ('conv2', GCNConv(16, 3))]
    named_children = list(module.named_children())

    # it builds the path of the structure
    # for instance, ('GCN', 'conv1', 'lin')
    path = path + (name, )

    # create namedtuple [path, (whether has) leaf, module]
    yield Trace(path, len(named_children) == 0, module)

    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from _walk_modules(child_module, name=name, path=path)


def format_time(time_us):
    """Defines how to format time in torch profiler Event.
    """
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return '{:.3f}s'.format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return '{:.3f}ms'.format(time_us / US_IN_MS)
    return '{:.3f}us'.format(time_us)


def format_memory(nbytes):
    """Returns a formatted memory size string in torch profiler Event
    """
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} Gb'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} Mb'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} Kb'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'
