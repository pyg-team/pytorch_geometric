"""Ranks relational candidates with a G-Retriever that fuses the
KumoRelational foundation model into Nemotron-H: KumoRelational's embedding of
each candidate is spliced into the prompt at the candidate's row, and the LLM
hidden state there scores the candidate. That score, the ``residual``,
corrects KumoRelational's churn logit, ``p' = sigmoid(logit p + lam *
residual)``, and candidates are then ranked by KumoRelational's utility
``U2`` under ``p'``, so ``--lam 0`` reproduces KumoRelational alone,
``--arm nemotron`` ranks by the LLM score alone and ``--arm rule`` by the
value score alone, as if every candidate churned. For ``P2`` questions,
``U2`` is the churn probability times the question's value score; for the
other families, it is the value score of the candidates predicted to churn
(with a probability of at least 0.5), followed by the other eligible
candidates by expected value. The benchmark comes from
``kumo_gretriever_data.py``.

KumoRelational stays frozen; ``--train`` fits the projector, the readout and
the full-precision LLM with a listwise cross-entropy over the gold set, and
``--train --finetune`` then refits them as the churn correction, with a
binary cross-entropy of ``sigmoid(logit p + residual)`` against each
candidate's churn label plus ``0.1 * residual ** 2``, which keeps the
correction small where the LLM has nothing to add (``--lam 0.75`` then
shrinks the correction it fits). Only finetuned checkpoints are meant for
``--arm fusion`` evaluation. The
graph tokens come from the model's own KumoRelational, which ``--save`` and
``--ckpt`` carry, and ``U2`` from a separate one: the pretrained model, or a
KumoRelational adapted outside this script whose :class:`KumoEncoder` state
dict ``--kumo-ckpt`` loads, so ``--ckpt`` never changes ``U2``. Pass
``--finetune`` the ``--kumo-ckpt`` of the evaluation, as the correction is fit
to the probabilities it corrects. Evaluation
needs about 40 GB of GPU memory (a few GB with ``--arm kumo``), training
about 165 GB (fp32 weights and AdamW state of the 9B LLM on a single GPU).

KumoRelational's in-context predictions depend on the generator seed (use
the ``--seed`` the checkpoints were trained with) and on the row order of
:obj:`sdm`'s relational joins, which neither cuDF's GPU hash join nor a
multi-threaded Arrow hash join keeps stable. The script therefore blocks cuDF
before :obj:`sdm` is imported (:obj:`sdm` then warns about its CPU fallbacks)
and runs Arrow single-threaded. The predictions are also sensitive to
rounding: they differ across GPU models (e.g., between GB200 and H100 GPUs),
and the script turns off TF32 matmuls, which NVIDIA containers turn on by
default. It also turns off cuDNN attention, which PyTorch prefers for
Nemotron-H's :obj:`bfloat16` attention on Hopper and Blackwell GPUs and which
changes the LLM scores; every reported evaluation ran without it.

Requirements: the NGC PyG container ``nvcr.io/nvidia/pyg:26.09-py3``, the
only tested environment (it provides PyTorch, PyG, RelBench and CUDA). Inside
it, run ``bash kumo_gretriever_setup.sh`` once. It installs
``structured-data-models`` (KumoRelational) and ``transformers==4.57.6``, and
builds the ``mamba-ssm`` and ``causal-conv1d`` kernels of Nemotron-H from
source with a one-line patch: their ``setup.py`` compiles with
``-std=c++17``, while the container's PyTorch 2.14 requires C++20. The patch
is needed until a newer container ships C++20-ready builds.

The KumoRelational checkpoint adapted to ``rel-amazon`` that ``--kumo-ckpt``
loads for the reported numbers is not published; please contact the author
(@puririshi98 on GitHub) for access. Without it, ``U2`` comes from the
pretrained KumoRelational. It and every other reported checkpoint were trained
only on questions anchored from 2008-04-10 to 2012-07-05, whose labels end on
2012-10-04, before the first test anchor.

Usage (``--as-of test`` stands for the 12 test anchors from 2013-01-03 to
2015-10-01, ``--train-as-of train`` for the 12 earlier ones from 2010-01-07 to
2012-10-04; the reported numbers are the mean of ``--seed 50`` and ``52``,
each with its own checkpoints)::

    python kumo_gretriever.py --arm kumo --kumo-ckpt kumo.pt --as-of test
    python kumo_gretriever.py --arm fusion --train --train-as-of train \
        --kumo-ckpt kumo.pt --save stage1.pt
    python kumo_gretriever.py --arm fusion --train --finetune \
        --ckpt stage1.pt --train-as-of train --kumo-ckpt kumo.pt \
        --save fusion.pt
    python kumo_gretriever.py --arm fusion --ckpt fusion.pt \
        --kumo-ckpt kumo.pt --as-of test
    python kumo_gretriever.py --arm nemotron --train --train-as-of train \
        --save nemotron.pt

The reported stage-one and Nemotron checkpoints were trained longer than these
commands do by default (about 2,000 updates, with an earlier version of this
script), and the
reported fusion checkpoints were finetuned from them as above.
"""
import argparse
import bisect
import copy
import json
import math
import random
import sys
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from kumo_gretriever_data import (
    ANCHORS,
    MARKER,
    build_benchmark,
    dec,
    eligible,
    offer_cost,
    score,
    strict_recall,
)
from torch import Tensor

from torch_geometric.llm.models import LLM, GRetriever

MODEL = 'nvidia/NVIDIA-Nemotron-Nano-9B-v2'
REVISION = '6533e8de2c68e4536bf7c411d7a3ce5734111476'
SYSTEM = ('Use the supplied candidate observations and relational '
          'representations to answer the question. Return only the requested '
          'candidate IDs, copied exactly, in best-first order as '
          'comma-separated values.\n/no_think')
INPUTS = ('x_context', 'y_context', 'x_query', 'related_context_tables',
          'related_query_tables')
# The selection rule of each question family, see `utility`:
RULES = dict(P2='expected_value', P5='predicted_churners',
             RV1='predicted_churners', RV3='predicted_churners')
EPS = 1e-6  # Clamps churn probabilities before their logit.


class KumoOutput(NamedTuple):
    candidate_ids: Tuple[str, ...]
    embeddings: Tensor  # [num_candidates, out_channels]
    probabilities: Tensor  # [num_candidates, num_classes]
    classes: Tuple[str, ...]


class KumoEncoder(torch.nn.Module):
    r"""Encodes candidates with a frozen :class:`sdm.models.KumoRelational`:
    the query states entering its in-context prediction head, averaged over
    seeded estimators, are the candidate embeddings.
    """
    def __init__(self, model: torch.nn.Module, num_estimators: int = 8,
                 num_hops: int = 2, seed: int = 50) -> None:
        super().__init__()
        if list(model.models) != ['classification']:
            raise ValueError("Expected KumoRelational(task='classification')")
        self.model = model.requires_grad_(False).eval()
        self.num_estimators = num_estimators
        self.num_hops, self.seed = num_hops, seed
        self.out_channels = self.head[0].in_features

    @property
    def head(self) -> torch.nn.Module:
        return self.model.models['classification'].icl_block.head

    def train(self, mode: bool = True) -> 'KumoEncoder':
        super().train(mode)
        self.model.eval()
        return self

    def predict_and_encode(self, pool: Dict[str, Any]) -> KumoOutput:
        ids, inputs = tuple(pool['candidate_ids']), pool['model_input']
        if (set(inputs) != set(INPUTS) or len(set(ids)) != len(ids)
                or inputs['x_query'].size(-2) != len(ids)):
            raise ValueError('Expected one query row per unique candidate')
        tables = [inputs['x_context'], inputs['x_query']]
        tables += [t for k in INPUTS[3:] for t in inputs[k].tables.values()]
        targets = set(inputs['y_context'].column_names)
        if any(targets & set(table.column_names) for table in tables):
            raise ValueError('Target columns cannot enter the features')

        states: List[Tensor] = []

        def capture(_: torch.nn.Module, args: Tuple[Tensor, ...]) -> None:
            if args[0].shape[-2:] != (len(ids), self.out_channels):
                raise ValueError('Unexpected query layout at the head')
            states.append(args[0].detach().clone().view(
                -1, *args[0].shape[-2:]))

        handle = self.head.register_forward_pre_hook(capture)
        try:  # Preprocessing may mutate inputs, so pass a fresh copy:
            device = next(self.model.parameters()).device
            generator = torch.Generator(device).manual_seed(self.seed)
            out = self.model(**copy.deepcopy(inputs), generator=generator,
                             num_estimators=self.num_estimators,
                             num_hops=self.num_hops)
        finally:
            handle.remove()
        # The class order is that of the numerical block (`column_names` is
        # an unordered set):
        classes = tuple(out.columns['numerical'])
        if (sum(s.size(0) for s in states) != self.num_estimators
                or out.numerical.shape != (len(ids), len(classes))):
            raise ValueError('Expected one output per candidate and estimator')
        if set(classes) != {'0', '1'}:
            raise ValueError(f'Expected binary churn classes, got {classes}')
        with torch.inference_mode(False):  # Leave sdm's inference tensors:
            embeddings = torch.cat(states).mean(dim=0).clone()
            probabilities = out.numerical.detach().clone()
        return KumoOutput(ids, embeddings, probabilities, classes)

    def forward(self, pool: Dict[str, Any]) -> Tensor:
        return self.predict_and_encode(pool).embeddings


class Readout(torch.nn.Module):
    r"""Scores one LLM hidden state per candidate, gated by :obj:`w`, which
    starts at zero so that an untrained model ranks like KumoRelational.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.up = torch.nn.Linear(in_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)
        self.w = torch.nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor) -> Tensor:
        out = self.out(F.gelu(self.up(self.norm(hidden.float()))))
        return self.w * out.squeeze(-1)


class RankGRetriever(GRetriever):
    r"""A :class:`~torch_geometric.llm.models.GRetriever` that ranks the
    candidates in its context instead of generating their ids: every marker
    takes the projected embedding of one candidate, and the LLM hidden state
    there is scored by :class:`Readout` (without an encoder: at the token
    closing each candidate's row). The prompt is assembled here rather than
    by :meth:`LLM._get_embeds` because the question precedes the context and
    one projected row is spliced per marker.
    """
    def __init__(self, llm: LLM, gnn: Optional[KumoEncoder] = None) -> None:
        super().__init__(llm, gnn)
        self.readout = Readout(llm.llm.config.hidden_size).to(llm.device)
        if gnn is None:  # Nothing to gate against:
            self.readout.w.data.fill_(1.0)
            self.readout.w.requires_grad_(False)

    def encode(self, pool: Dict[str, Any]) -> Tensor:
        return self.gnn(pool)  # One row per candidate, no pooling.

    def _embed(self, question: str, context: str,
               xs: Optional[Tensor]) -> Tuple[Tensor, Tensor, List[int]]:
        tokenizer, device = self.llm.tokenizer, self.llm.device
        if xs is None:  # Text only: score every candidate at its row's end.
            context = context.replace(MARKER, '')
        messages = [
            dict(role='system', content=self.llm.sys_prompt),
            dict(role='user', content=f'{question}\n\nCandidates:\n{context}'),
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        bos = tokenizer.bos_token or ''
        bos = bos if text.startswith(bos) else ''
        ids = tokenizer(bos, add_special_tokens=False).input_ids if bos else []
        text, seq, positions, cur = text[len(bos):], [], [], 0
        if xs is None:
            enc = tokenizer(text, add_special_tokens=False,
                            return_offsets_mapping=True)
            positions = [
                len(ids) + _token_at(enc.offset_mapping, end)
                for end in _row_ends(text, context)
            ]
            ids += enc.input_ids
            seq.append(self.word_embedding(torch.tensor(ids, device=device)))
        else:
            parts = text.split(MARKER)
            if len(parts) != xs.size(0) + 1:
                raise ValueError('Expected one candidate marker per embedding')
            for i, part in enumerate(parts):  # Tokenized piece by piece:
                ids += tokenizer(part, add_special_tokens=False).input_ids
                if len(ids) > 0:
                    seq.append(
                        self.word_embedding(torch.tensor(ids, device=device)))
                    cur, ids = cur + len(ids), []
                if i < xs.size(0):
                    positions.append(cur)
                    seq.append(xs[i:i + 1])
                    cur += 1
        inputs_embeds = torch.cat(seq).unsqueeze(0)
        mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                          device=device)
        return inputs_embeds, mask, positions

    def forward(self, question: str, context: str,
                pool: Optional[Dict[str, Any]] = None,
                x: Optional[Tensor] = None) -> Tensor:
        r"""Scores the candidates of :obj:`context` from their encoded
        :obj:`pool` (or already encoded :obj:`x`); text only without either.
        """
        if x is None and pool is not None:
            x = self.encode(pool)
        xs = None
        if x is not None:
            xs = self._align_dtype(self.projector(x), self.llm_generator)
            xs = xs.flatten(0, 1)
        inputs_embeds, mask, positions = self._embed(question, context, xs)
        base = self.llm_generator  # Nemotron-H keeps its base in `backbone`:
        base = base.backbone if hasattr(base, 'backbone') else base.model
        with self.llm.autocast_context:
            hidden = base(inputs_embeds=inputs_embeds, attention_mask=mask,
                          use_cache=False, return_dict=True).last_hidden_state
        return self.readout(hidden[0, positions])

    def inference(self, *args: Any, **kwargs: Any) -> List[str]:
        raise NotImplementedError(f"'{self.__class__.__name__}' ranks "
                                  "candidates in 'forward' and does not "
                                  "generate answers")


def _row_ends(text: str, context: str) -> List[int]:
    r"""Character index of the closing brace of every candidate row."""
    decoder, ends, cursor = json.JSONDecoder(), [], 0
    for line in filter(None, map(str.strip, context.split('\n'))):
        row = line[:decoder.raw_decode(line)[1]]
        cursor = text.index(row, cursor) + len(row)
        ends.append(cursor - 1)
    return ends


def _token_at(offsets: List[Tuple[int, int]], char: int) -> int:
    r"""Index of the token whose span contains :obj:`char`."""
    t = bisect.bisect_right([start for start, _ in offsets], char) - 1
    while t >= 0 and offsets[t][0] == offsets[t][1]:  # Skip empty spans.
        t -= 1
    if t < 0 or not offsets[t][0] <= char < offsets[t][1]:
        raise ValueError('Candidate row end is not inside a token')
    return t


def utility(candidates: List[dict], policy: dict,
            probabilities: Optional[dict] = None,
            rule: Optional[str] = None) -> List[Optional[Decimal]]:
    r"""KumoRelational's utility :obj:`U2` of every eligible candidate under
    :obj:`rule`, by default the family's rule in :obj:`RULES`: the expected
    value :obj:`score * p(churn)` for :obj:`"expected_value"`, or the score of
    a predicted churner (:obj:`p(churn) >= 0.5`) for
    :obj:`"predicted_churners"`. It is :obj:`0` without probabilities and
    :obj:`None` for the candidates left out.
    """
    rule = RULES[policy['family']] if rule is None else rule
    if rule not in ['expected_value', 'predicted_churners']:
        raise ValueError(f"Unknown selection rule '{rule}'")
    utilities: List[Optional[Decimal]] = []
    for c in candidates:
        if not eligible(c, policy):
            utilities.append(None)
        elif probabilities is None:
            utilities.append(Decimal(0))
        elif rule == 'expected_value':
            utilities.append(
                score(c['features'], policy) * dec(probabilities[c['id']]))
        elif dec(probabilities[c['id']]) >= Decimal('0.5'):
            utilities.append(score(c['features'], policy))
        else:  # Not a predicted churner.
            utilities.append(None)
    return utilities


def spread(utilities: List[Optional[Decimal]]) -> float:
    r"""Population standard deviation, skipping :obj:`None` utilities."""
    values = [float(u) for u in utilities if u is not None]
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean)**2 for v in values) / len(values))
    return std if std > 1e-12 and math.isfinite(std) else 1.0


def tiered_utility(
        candidates: List[dict], policy: dict,
        probabilities: Optional[dict] = None, rule: Optional[str] = None
) -> Tuple[List[int], List[Optional[Decimal]]]:
    r"""The tier and the utility of :func:`rank`: tier 0 holds the
    candidates :obj:`rule` selects, under :obj:`U2`, tier 1 the other
    eligible candidates, under the expected value.
    """
    utilities = utility(candidates, policy, probabilities, rule)
    fill = utility(candidates, policy, probabilities, 'expected_value')
    tiers = [0 if u is not None else 1 for u in utilities]
    return tiers, [u if u is not None else f for u, f in zip(utilities, fill)]


def rank(candidates: List[dict], policy: dict,
         probabilities: Optional[dict] = None,
         residual: Optional[List[float]] = None, lam: float = 0.75,
         rule: Optional[str] = None) -> List[str]:
    r"""Best-first candidate ids under :obj:`U2` from :func:`utility` under
    :obj:`rule`, with the churn probabilities corrected to
    :obj:`sigmoid(logit p + lam * residual)` (:obj:`lam=0` is KumoRelational
    alone; without probabilities, the residual alone). The eligible
    candidates that :obj:`rule` leaves out follow, under the expected value
    in place of :obj:`U2` (:func:`tiered_utility`). The list is cut to the
    requested count and, for RV3, to the budget, or shorter if fewer
    candidates are eligible.
    """
    residual = residual if residual is not None else [0.0] * len(candidates)
    if probabilities is not None:  # The residual corrects the churn logit:
        corrected = {}
        for c, r in zip(candidates, residual):
            if not math.isfinite(float(r)):
                raise ValueError('The residual is not finite')
            p = min(max(float(probabilities[c['id']]), EPS), 1 - EPS)
            z = math.log(p / (1 - p)) + lam * float(r)
            corrected[c['id']] = (1 / (1 + math.exp(-z)) if z >= 0 else
                                  math.exp(z) / (1 + math.exp(z)))
        probabilities, residual = corrected, [0.0] * len(candidates)
    tiers, utilities = tiered_utility(candidates, policy, probabilities, rule)
    keys = {}
    for c, t, u, r in zip(candidates, tiers, utilities, residual):
        if u is not None:
            keys[c['id']] = (t, -(u + Decimal(r)),
                             -dec(c['features']['prior_ltv']), int(c['id']))
    features = {c['id']: c['features'] for c in candidates}
    ids, spent = [], Decimal(0)
    for i in sorted(keys, key=keys.get):
        if policy['family'] == 'RV3':
            spent += offer_cost(features[i], policy)
            if spent > dec(policy['budget']):
                break
        ids.append(i)
        if len(ids) == policy['requested_count']:
            break
    return ids


def listwise_loss(base: Tensor, residual: Tensor,
                  gold: Tensor) -> Optional[Tensor]:
    r"""Softmax cross-entropy of the gold set under :obj:`base + residual`,
    with ineligible candidates at :obj:`-inf`.
    """
    gold = gold & base.isfinite()
    if not bool(gold.any()):
        return None
    return -torch.log_softmax(base + residual, dim=-1)[gold].mean()


def build_llm(model_name: str = MODEL, revision: str = REVISION,
              device: Any = 'cuda:0') -> LLM:
    r"""Wraps Nemotron-H as an :class:`~torch_geometric.llm.models.LLM` with
    full-precision weights, eager attention and :obj:`bfloat16` autocast.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 trust_remote_code=True,
                                                 dtype=torch.float32,
                                                 device_map={'': device},
                                                 attn_implementation='eager')
    model.config.use_cache = False
    llm = LLM.__new__(LLM)  # The wrapper's loader assumes Llama-like models.
    torch.nn.Module.__init__(llm)
    llm.model_name, llm.tokenizer, llm.llm = model_name, tokenizer, model
    llm.word_embedding = model.get_input_embeddings()
    llm.device, llm.sys_prompt = torch.device(device), SYSTEM
    llm.autocast_context = torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return llm


def predict(
    model: Optional[RankGRetriever],
    encoder: Optional[KumoEncoder],
    record: Dict[str, Any],
) -> Tuple[Optional[Dict[str, float]], Optional[Tensor]]:
    r"""The churn probability by candidate id from :obj:`encoder` and the
    residual of a record from :obj:`model`, whose own GNN encodes the graph
    tokens.
    """
    probabilities = residual = None
    if encoder is not None:
        out = encoder.predict_and_encode(record['pool'])
        p = out.probabilities[:, out.classes.index('1')].clamp(0, 1)
        probabilities = dict(zip(out.candidate_ids, p.tolist()))
    if model is not None:
        pool = record['pool'] if model.gnn is not None else None
        residual = model(record['question'], record['context'], pool)
    return probabilities, residual


def train(model: RankGRetriever, encoder: Optional[KumoEncoder],
          records: List[Dict[str, Any]], epochs: int, seed: int,
          finetune: bool = False) -> None:
    r"""Fits the residual with a listwise cross-entropy over the gold set.
    With :obj:`finetune`, it fits the residual as the correction of the churn
    logit that :func:`rank` applies, by binary cross-entropy against the
    churn labels, with a :obj:`0.1 * residual ** 2` penalty.
    """
    groups = [dict(params=list(model.llm_generator.parameters()), lr=1e-5)]
    if model.gnn is not None:
        groups.append(dict(params=list(model.projector.parameters()), lr=1e-4))
    head = [p for name, p in model.readout.named_parameters() if name != 'w']
    groups.append(dict(params=head, lr=1e-3))
    if model.readout.w.requires_grad:
        groups.append(dict(params=[model.readout.w], lr=1e-2, weight_decay=0))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    model.llm_generator.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.train()
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        rng.shuffle(records)
        total = 0.0
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            optimizer.zero_grad(set_to_none=True)
            for record in batch:
                probabilities, residual = predict(model, encoder, record)
                if finetune:  # The churn correction of rank():
                    labels = {str(k): v for k, v in record['labels'].items()}
                    ids = [c['id'] for c in record['candidates']]
                    p = torch.tensor([
                        min(max(float(probabilities[i]), EPS), 1 - EPS)
                        for i in ids
                    ], device=residual.device)
                    y = torch.tensor([float(labels[str(i)]) for i in ids],
                                     device=residual.device)
                    residual = residual.float()
                    loss = F.binary_cross_entropy_with_logits(
                        torch.log(p / (1 - p)) + residual, y)
                    loss = loss + 0.1 * (residual**2).mean()
                else:
                    # The base is the expected value for every family, as the
                    # first stage was trained; for P5, whose scores are
                    # negative, it ranks likelier churners lower:
                    gold = set(record['gold'])
                    mask = torch.tensor(
                        [c['id'] in gold for c in record['candidates']],
                        device=residual.device)
                    utilities = utility(record['candidates'], record['policy'],
                                        probabilities, 'expected_value')
                    s = spread(utilities)
                    base = [
                        -math.inf if u is None else float(u) / s
                        for u in utilities
                    ]
                    loss = listwise_loss(
                        torch.tensor(base, device=residual.device), residual,
                        mask)
                if loss is not None:
                    (loss / len(batch)).backward()
                    total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f'Epoch {epoch}: loss {total / len(records):.4f}')


def anchors(names: List[str]) -> List[str]:
    r"""Expands ``test`` and ``train`` to the 12 test anchors and the 12
    anchors before them.
    """
    groups = dict(test=list(ANCHORS[20:]), train=list(ANCHORS[8:20]))
    return [a for name in names for a in groups.get(name, [name])]


@torch.no_grad()
def evaluate(model: Optional[RankGRetriever], encoder: Optional[KumoEncoder],
             records: List[dict], arm: str, lam: float) -> None:
    recall: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        probabilities, residual = predict(model, encoder, record)
        if arm == 'rule':  # As if every candidate churned:
            probabilities = dict.fromkeys(record['pool']['candidate_ids'], 1.0)
        if residual is not None:
            residual = residual.double().tolist()
        ids = rank(record['candidates'], record['policy'], probabilities,
                   residual, lam)
        by_anchor = recall.setdefault(record['task'], {})
        by_anchor.setdefault(record['anchor'], []).append(
            strict_recall(ids, record['gold'], record['pool']['candidate_ids'],
                          record['k']))
    for task, by_anchor in recall.items():
        means = []
        for anchor, values in by_anchor.items():
            means.append(sum(values) / len(values))
            print(f'{arm} ({task}, {anchor}): recall {means[-1]:.4f} over '
                  f'{len(values)} questions')
        print(f'{arm} ({task}): recall {sum(means) / len(means):.4f} '
              f'(anchor mean, n={len(means)})')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='fusion',
                        choices=['fusion', 'kumo', 'nemotron', 'rule'])
    parser.add_argument('--lam', type=float, default=0.75,
                        help='The residual weight')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--root',
                        default='stanford-star/relbench-v1/rel-amazon')
    parser.add_argument(
        '--as-of', nargs='+', default=['2015-10-01'],
        help='Test anchors, or `test` for all 12 from 2013-01-03 to '
        '2015-10-01, 91 days apart')
    parser.add_argument(
        '--train-as-of', nargs='+', default=['2012-10-04'],
        help='Training anchors, or `train` for the 12 before the test ones')
    parser.add_argument('--num-questions', type=int, default=8)
    parser.add_argument('--train-questions', type=int, default=48,
                        help='Training questions per anchor and task')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--finetune', action='store_true',
                        help='Train again as the churn correction')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ckpt', help='The RankGRetriever state dict to load')
    parser.add_argument('--kumo-ckpt', help='The KumoEncoder state dict of U2')
    parser.add_argument('--save', help='Where to save the trained state dict')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.as_of = anchors(args.as_of)
    args.train_as_of = anchors(args.train_as_of)
    if args.arm in ['kumo', 'rule'] and (args.ckpt or args.train):
        parser.error(f'--arm {args.arm} takes no --ckpt or --train')
    if args.arm in ['nemotron', 'rule'] and args.kumo_ckpt:
        parser.error(f'--arm {args.arm} takes no --kumo-ckpt')
    if args.save and not args.train:
        parser.error('--save needs --train')
    if args.finetune and not (args.train and args.ckpt):
        parser.error('--finetune needs --train and --ckpt')
    if args.finetune and args.arm != 'fusion':
        parser.error('--finetune needs --arm fusion')
    if args.train and set(args.as_of) & set(args.train_as_of):
        parser.error('--as-of and --train-as-of overlap')
    # Arrow joins instead of cuDF, no TF32 and no cuDNN attention, see the
    # docstring. Arrow runs single-threaded since its multi-threaded hash join
    # reorders the graph's edges, and `temporal_strategy='last'` then samples
    # other neighbors at time ties:
    sys.modules['cudf'] = None
    import pyarrow
    pyarrow.set_cpu_count(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    records = build_benchmark(args.root, args.as_of,
                              n_questions=args.num_questions, seed=args.seed)
    train_records = []
    if args.train:  # From their own stream, apart from the test questions:
        train_records = build_benchmark(args.root, args.train_as_of,
                                        n_questions=args.train_questions,
                                        seed=args.seed)
    for record in records + train_records:
        record['pool']['model_input'] = {
            key: value.to(device)
            for key, value in record['pool']['model_input'].items()
        }

    encoder = model = None
    if args.arm in ['fusion', 'kumo']:
        from sdm.models import KumoRelational
        kumo = KumoRelational(task='classification', device=device)
        encoder = KumoEncoder(kumo, seed=args.seed)  # Scores `U2`.
        if args.kumo_ckpt is not None:
            encoder.load_state_dict(
                torch.load(args.kumo_ckpt, map_location='cpu',
                           weights_only=True, mmap=True))
    if args.arm in ['fusion', 'nemotron']:
        gnn = None
        if encoder is not None:  # Its own copy encodes the graph tokens:
            kumo = KumoRelational(task='classification', device=device)
            gnn = KumoEncoder(kumo, seed=args.seed)
        model = RankGRetriever(build_llm(device=device), gnn)
        if args.ckpt is not None:
            model.load_state_dict(
                torch.load(args.ckpt, map_location='cpu', weights_only=True,
                           mmap=True))
        if args.train:
            train(model, encoder, train_records, args.epochs, args.seed,
                  args.finetune)
            if args.save is not None:
                torch.save(model.state_dict(), args.save)
            if args.arm == 'fusion' and not args.finetune:
                print('Stage one done; evaluate after --finetune')
                return
        # Frozen weights keep autocast from caching a bf16 copy of the LLM:
        model.eval().requires_grad_(False)
    evaluate(model, encoder, records, args.arm, args.lam)


if __name__ == '__main__':
    main()
