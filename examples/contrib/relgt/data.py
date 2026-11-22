"""Small, CPU-only temporal context builder for the RelGT example.

This module has no RelBench or TorchFrame dependency. It accepts an audited
row graph and returns fixed-size contexts; loading a database and determining
when its attributes became available remain the caller's responsibility.
Unknown timestamps are admitted only through ``time_known=False``, which is
an explicit caller assertion that the row is safe to treat as static.

The implementation prioritizes inspectable behavior for small fixtures. It
rebuilds adjacency and fingerprints the graph per request; large databases
need a separately validated preprocessing adapter. It does not establish
the model's global-memory fit-origin or label-availability guarantees.
"""

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

SCHEMA_VERSION = 1
SAMPLER_VERSION = 'relgt-temporal-v1'


@dataclass(frozen=True)
class RelGTGraph:
    """A row graph with global row indices and type-local source IDs.

    All tensors reside on CPU. ``time`` is int64 Unix seconds. An edge is
    available when both endpoint rows are available; the caller must filter
    separately dated or mutable relationships before constructing this graph.
    ``node_type`` and ``node_id`` jointly identify a unique database row.
    """
    node_type: Tensor
    node_id: Tensor
    time: Tensor
    time_known: Tensor
    edge_index: Tensor

    def validate(self) -> None:
        n = self.node_type.numel()
        for name in ('node_type', 'node_id', 'time', 'time_known'):
            value = getattr(self, name)
            dtype = torch.bool if name == 'time_known' else torch.long
            if (value.device.type != 'cpu' or value.dtype != dtype
                    or value.shape != (n, )):
                raise ValueError(f'{name} must be a CPU {dtype} vector')
        if (self.node_type < 0).any() or (self.node_id < 0).any():
            raise ValueError('Node types and source IDs must be nonnegative')
        keys = list(zip(self.node_type.tolist(), self.node_id.tolist()))
        if len(set(keys)) != n:
            raise ValueError('Type and source ID must identify unique rows')
        edge = self.edge_index
        if (edge.device.type != 'cpu' or edge.dtype != torch.long
                or edge.dim() != 2 or edge.size(0) != 2):
            raise ValueError('edge_index must be a CPU int64 [2, E] tensor')
        if edge.numel() and ((edge < 0).any() or (edge >= n).any()):
            raise ValueError('edge_index contains an out-of-range row')


@dataclass(frozen=True)
class PredictionSample:
    """A task row, distinct from its possibly repeated seed entity.

    Revisions must identify immutable data/task artifacts. ``task_row_id``
    is assigned before sorting or filtering the pinned split. Timestamps
    are integer Unix seconds. ``seed_index`` is a global row index.
    """
    dataset_revision: str
    task_revision: str
    split: str
    task_row_id: str
    seed_index: int
    prediction_time: int


@dataclass(frozen=True)
class RelGTContext:
    """One context; repeated entities retain distinct token positions."""
    sample_id: str
    node_index: Tensor
    node_type: Tensor
    node_id: Tensor
    hop: Tensor
    age: Tensor
    edge_index: Tensor


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _digest(metadata: object, tensors: Dict[str, Tensor]) -> str:
    digest = hashlib.sha256(_json_bytes(metadata))
    for name, tensor in sorted(tensors.items()):
        metadata = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(_json_bytes(metadata))
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sample_id(graph: RelGTGraph, sample: PredictionSample) -> str:
    for name in ('dataset_revision', 'task_revision', 'split', 'task_row_id'):
        value = getattr(sample, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'{name} must be a nonempty string')
    if (type(sample.seed_index) is not int
            or not 0 <= sample.seed_index < graph.node_id.numel()):
        raise ValueError('seed_index is not a valid global row index')
    if type(sample.prediction_time) is not int:
        raise ValueError('prediction_time must be integer Unix seconds')
    key = dict(vars(sample))
    key['node_type'] = int(graph.node_type[sample.seed_index])
    key['node_id'] = int(graph.node_id[sample.seed_index])
    return hashlib.sha256(_json_bytes(key)).hexdigest()


def _adjacency(graph: RelGTGraph) -> List[Set[int]]:
    adjacency: List[Set[int]] = [set() for _ in range(graph.node_id.numel())]
    for src, dst in graph.edge_index.t().tolist():
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    return adjacency


def _occurrence_edges(nodes: List[int], adjacency: List[Set[int]]) -> Tensor:
    positions: Dict[int, List[int]] = {}
    for position, node in enumerate(nodes):
        positions.setdefault(node, []).append(position)
    edges = set()
    for src, node in enumerate(nodes):
        for neighbor in adjacency[node]:
            for dst in positions.get(neighbor, []):
                edges.add((src, dst))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def _context_tensors(context: RelGTContext) -> Dict[str, Tensor]:
    return {
        name: getattr(context, name)
        for name in ('node_index', 'node_type', 'node_id', 'hop', 'age',
                     'edge_index')
    }


def _validate_context(context: RelGTContext, graph: RelGTGraph,
                      sample: PredictionSample, sample_id: str, size: int,
                      adjacency: List[Set[int]], eligible: Tensor) -> None:
    if context.sample_id != sample_id:
        raise ValueError('Cached context has an incompatible sample identity')
    for name in ('node_index', 'node_type', 'node_id', 'hop'):
        value = getattr(context, name)
        if (value.device.type != 'cpu' or value.dtype != torch.long
                or value.shape != (size, )):
            raise ValueError(f'Invalid cached {name}')
    nodes = context.node_index
    if ((nodes < 0).any() or (nodes >= graph.node_id.numel()).any()
            or nodes[0] != sample.seed_index):
        raise ValueError('Invalid cached node indices or seed position')
    if not eligible[nodes].all():
        raise ValueError('Cached context contains a future row')
    if (not torch.equal(context.node_type, graph.node_type[nodes])
            or not torch.equal(context.node_id, graph.node_id[nodes])):
        raise ValueError('Cached node identity does not match the graph')
    if (context.hop[0] != 0 or (context.hop[1:] < 1).any()
            or (context.hop[1:] > 3).any()):
        raise ValueError('Invalid cached hop metadata')
    if (context.age.device.type != 'cpu' or context.age.dtype != torch.float32
            or context.age.shape != (size, )):
        raise ValueError('Invalid cached age tensor')
    age = torch.where(graph.time_known[nodes],
                      (sample.prediction_time - graph.time[nodes]).double() /
                      86400, 0).float()
    age[0] = 0
    if not torch.equal(context.age, age):
        raise ValueError('Cached ages do not match the prediction time')
    edges = _occurrence_edges(nodes.tolist(), adjacency)
    if (context.edge_index.dtype != torch.long
            or context.edge_index.device.type != 'cpu'
            or not torch.equal(context.edge_index, edges)):
        raise ValueError('Cached occurrence edges do not match the graph')


def _write_cache(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
            torch.save(payload, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def sample_context(
    graph: RelGTGraph,
    sample: PredictionSample,
    context_size: int = 300,
    seed: int = 0,
    max_one_hop: int = 5000,
    max_two_hop: int = 1000,
    cache_dir: Optional[Path] = None,
) -> RelGTContext:
    """Sample a deterministic temporal context, optionally caching it.

    Timed nodes are eligible at or before the prediction time. Nonempty
    small neighborhoods use replacement. Isolated seeds use eligible global
    rows; if none exist, the seed is repeated. The cache is keyed by task row
    and validates graph/configuration hashes before reuse. An incompatible
    existing entry raises ``ValueError`` instead of silently overwriting it.
    """
    graph.validate()
    for name, value in [('context_size', context_size),
                        ('max_one_hop', max_one_hop),
                        ('max_two_hop', max_two_hop)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    sample_id = _sample_id(graph, sample)
    config = dict(version=SAMPLER_VERSION, context_size=context_size,
                  seed=seed, max_one_hop=max_one_hop, max_two_hop=max_two_hop)
    fingerprint = _digest(config, vars(graph))
    eligible = ~graph.time_known | (graph.time <= sample.prediction_time)
    if not eligible[sample.seed_index]:
        raise ValueError('The seed row is unavailable at prediction time')
    adjacency = _adjacency(graph)
    cache_path = None if cache_dir is None else Path(cache_dir) / (sample_id +
                                                                   '.pt')
    if cache_path is not None and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location='cpu',
                                 weights_only=True)
            if (payload['schema_version'] != SCHEMA_VERSION
                    or payload['fingerprint'] != fingerprint):
                raise ValueError('Incompatible cache schema or fingerprint')
            context = RelGTContext(**payload['context'])
            if payload['checksum'] != _digest(context.sample_id,
                                              _context_tensors(context)):
                raise ValueError('Cached context checksum mismatch')
            _validate_context(context, graph, sample, sample_id, context_size,
                              adjacency, eligible)
            return context
        except Exception as exc:
            raise ValueError(f'Invalid context cache: {cache_path}') from exc

    rng_seed = hashlib.sha256(_json_bytes([sample_id, config])).digest()
    rng = random.Random(int.from_bytes(rng_seed, 'big'))

    def ordered(nodes: Set[int]) -> List[int]:
        return sorted(
            nodes, key=lambda i:
            (int(graph.node_type[i]), int(graph.node_id[i])))

    def capped(nodes: Set[int], cap: int) -> Set[int]:
        candidates = ordered({i for i in nodes if eligible[i]})
        return set(
            rng.sample(candidates, cap) if len(candidates) >
            cap else candidates)

    first_full = {
        i
        for i in adjacency[sample.seed_index]
        if i != sample.seed_index and eligible[i]
    }
    first = capped(first_full, max_one_hop)
    second: Set[int] = set()
    for node in ordered(first):
        second.update(capped(adjacency[node], max_two_hop))
    second -= first | {sample.seed_index}
    candidates = [(i, 1) for i in ordered(first)]
    candidates += [(i, 1 if i in first_full else 2) for i in ordered(second)]
    count = context_size - 1
    if not candidates:
        fallback = {
            i
            for i in range(eligible.numel())
            if eligible[i] and i != sample.seed_index
        }
        candidates = [(i, 3) for i in ordered(fallback)]
    if not candidates:
        candidates = [(sample.seed_index, 3)]
    chosen: List[Tuple[int, int]] = (rng.sample(candidates, count) if
                                     len(candidates) >= count else rng.choices(
                                         candidates, k=count))
    rng.shuffle(chosen)
    nodes = torch.tensor([sample.seed_index] + [i for i, _ in chosen])
    hops = torch.tensor([0] + [hop for _, hop in chosen])
    age = torch.where(graph.time_known[nodes],
                      (sample.prediction_time - graph.time[nodes]).double() /
                      86400, 0).float()
    age[0] = 0
    context = RelGTContext(sample_id, nodes, graph.node_type[nodes],
                           graph.node_id[nodes], hops, age,
                           _occurrence_edges(nodes.tolist(), adjacency))
    _validate_context(context, graph, sample, sample_id, context_size,
                      adjacency, eligible)
    if cache_path is not None:
        payload = dict(schema_version=SCHEMA_VERSION, fingerprint=fingerprint,
                       context=vars(context),
                       checksum=_digest(sample_id, _context_tensors(context)))
        _write_cache(cache_path, payload)
    return context


def collate_contexts(contexts: List[RelGTContext],
                     row_embeddings: Tensor) -> Dict[str, object]:
    """Gather row embeddings and offset disjoint occurrence graphs.

    Returns model inputs with ``x`` shaped ``[B, K, C]``. ``seed_id`` uses
    global graph row indices, not the type-local ``node_id``. Metadata moves
    to the embedding device; ``sample_id`` remains a list of stable strings.
    Gathering preserves gradients to trainable row embeddings.
    """
    if not contexts or row_embeddings.dim() != 2:
        raise ValueError('Need nonempty contexts and [N, C] row embeddings')
    size = contexts[0].node_index.numel()
    if any(ctx.node_index.numel() != size for ctx in contexts):
        raise ValueError('All contexts must have the same size')
    device = row_embeddings.device
    index = torch.stack([ctx.node_index for ctx in contexts]).to(device)
    if ((index < 0).any() or (index >= row_embeddings.size(0)).any()):
        raise ValueError('Context rows are outside the embedding table')
    edges = torch.cat(
        [ctx.edge_index + i * size for i, ctx in enumerate(contexts)],
        dim=1).to(device)
    batch: Dict[str,
                object] = dict(x=row_embeddings[index], seed_id=index[:, 0],
                               edge_index=edges,
                               sample_id=[ctx.sample_id for ctx in contexts])
    for name in ('node_type', 'hop', 'age'):
        batch[name] = torch.stack([getattr(ctx, name)
                                   for ctx in contexts]).to(device)
    return batch
