"""Predict future inactivity from simulated banking histories.

Customers own one or two accounts and generate dated transactions from a
stochastic activity process. The target is no recorded transaction during
the next 30 days. All input features use events strictly before prediction.
These invented histories demonstrate an application, not banking accuracy.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.contrib.nn.models import RelGT, RelGTTokenEncoder

from .data import (
    PredictionSample,
    RelGTGraph,
    collate_contexts,
    sample_context,
)

DAY = 86400
ORIGINS = {'train': 90, 'val': 150, 'test': 210}
HORIZON = 30


@dataclass
class BankingHistory:
    num_customers: int
    account_customer: Tensor
    transaction_account: Tensor
    transaction_day: Tensor
    transaction_amount: Tensor

    def graph(self) -> RelGTGraph:
        n, a = self.num_customers, self.account_customer.numel()
        m = self.transaction_day.numel()
        return RelGTGraph(
            node_type=torch.cat(
                [torch.zeros(n),
                 torch.ones(a),
                 torch.full((m, ), 2)]).long(), node_id=torch.cat(
                     [torch.arange(n),
                      torch.arange(a),
                      torch.arange(m)]), time=torch.cat([
                          torch.zeros(n + a, dtype=torch.long),
                          self.transaction_day * DAY + 1
                      ]), time_known=torch.ones(n + a + m, dtype=torch.bool),
            edge_index=torch.stack([
                torch.cat(
                    [self.account_customer, self.transaction_account + n]),
                torch.cat([torch.arange(a) + n,
                           torch.arange(m) + n + a]),
            ]))


def generate_history(seed: int = 0, num_customers: int = 96,
                     days: int = 240) -> BankingHistory:
    """Generate a ledger without encoding outcomes in IDs or row attributes.

    Activity switches stochastically between engaged and dormant states.
    States and transition probabilities are simulation internals, never model
    inputs. Low transaction rates and recovery permit uncertain outcomes.
    """
    if num_customers < 8 or days < max(ORIGINS.values()) + HORIZON:
        raise ValueError('Need at least 8 customers and 240 observed days')
    rng = torch.Generator().manual_seed(seed)
    account_counts = torch.randint(1, 3, (num_customers, ), generator=rng)
    account_customer = torch.arange(num_customers).repeat_interleave(
        account_counts)
    offsets = account_counts.cumsum(0) - account_counts
    rate = 0.12 + 0.20 * torch.rand(num_customers, generator=rng)
    engaged = torch.rand(num_customers, generator=rng) > 0.2
    accounts, dates, amounts = [], [], []
    for day in range(days):
        switch = torch.rand(num_customers, generator=rng)
        engaged = torch.where(engaged, switch >= 0.004, switch < 0.002)
        active = torch.rand(num_customers, generator=rng) < torch.where(
            engaged, rate, torch.full_like(rate, 0.001))
        for customer in active.nonzero().flatten().tolist():
            choice = int(
                torch.randint(int(account_counts[customer]), (1, ),
                              generator=rng))
            accounts.append(int(offsets[customer]) + choice)
            dates.append(day)
            amount = float(torch.exp(4 + torch.randn((), generator=rng)))
            sign = -1 if float(torch.rand((), generator=rng)) < 0.7 else 1
            amounts.append(sign * amount)
    return BankingHistory(num_customers, account_customer,
                          torch.tensor(accounts, dtype=torch.long),
                          torch.tensor(dates, dtype=torch.long),
                          torch.tensor(amounts, dtype=torch.float))


def inactivity_target(history: BankingHistory, origin: int) -> Tensor:
    future = ((history.transaction_day >= origin)
              & (history.transaction_day < origin + HORIZON))
    owners = history.account_customer[history.transaction_account[future]]
    return (owners.bincount(minlength=history.num_customers) == 0).float()


def history_features(history: BankingHistory, origin: int) -> Tensor:
    """Use fixed units and past-only aggregates without future statistics."""
    n, a = history.num_customers, history.account_customer.numel()
    m = history.transaction_day.numel()
    features = torch.zeros(n + a + m, 8)
    owner = history.account_customer[history.transaction_account]
    past = history.transaction_day < origin
    for group, count, offset in [(owner, n, 0),
                                 (history.transaction_account, a, n)]:
        for index in range(count):
            selected = past & (group == index)
            days = history.transaction_day[selected]
            amounts = history.transaction_amount[selected]
            recent = days >= origin - 30
            prior = (days >= origin - 60) & (days < origin - 30)
            features[offset + index, :4] = torch.tensor([
                float(recent.sum()) / 10,
                float(prior.sum()) / 10,
                float(amounts[recent].abs().sum().log1p()) / 10,
                min(origin - int(days.max()), 60) / 60 if days.numel() else 1,
            ])
    features[:n, 4] = history.account_customer.bincount(minlength=n) / 2
    # Future transaction rows remain zero; they cannot enter a sampled context.
    rows = past.nonzero().flatten() + n + a
    value = history.transaction_amount[past]
    weekday = history.transaction_day[past].float().remainder(7)
    features[rows, 0] = value.abs().log1p() / 10
    features[rows, 1] = value.sign()
    features[rows, 2] = torch.sin(weekday * (2 * math.pi / 7))
    features[rows, 3] = torch.cos(weekday * (2 * math.pi / 7))
    return features


def prepare_split(history: BankingHistory, split: str, seed: int,
                  context_size: int = 32) -> Dict[str, Tensor]:
    origin, graph = ORIGINS[split], history.graph()
    contexts = [
        sample_context(
            graph,
            PredictionSample(f'simulated-bank-v2-{seed}',
                             'future-inactivity-30d-v1', split, str(i), i,
                             origin * DAY), context_size, seed)
        for i in range(history.num_customers)
    ]
    batch = collate_contexts(contexts, history_features(history, origin))
    batch['target'] = inactivity_target(history, origin)
    batch['node_index'] = torch.stack([c.node_index for c in contexts])
    return batch


def fixed_noise(split: str, seed: int, count: int, size: int) -> Tensor:
    # Each customer/timestamp pair has an independent reproducible stream.
    rows = []
    for customer in range(count):
        key = f'{seed}:{customer}:{ORIGINS[split]}'.encode()
        noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
        generator = torch.Generator().manual_seed(noise_seed)
        rows.append(torch.randn(size, 1, generator=generator))
    return torch.stack(rows)


def metrics(logits: Tensor, target: Tensor) -> dict:
    positive, negative = logits[target == 1], logits[target == 0]
    auc = None
    if positive.numel() and negative.numel():
        difference = positive[:, None] - negative[None, :]
        auc = float(((difference > 0).float() + 0.5 *
                     (difference == 0).float()).mean())
    return dict(
        bce=float(F.binary_cross_entropy_with_logits(logits, target)),
        roc_auc=auc, accuracy=float(((logits
                                      >= 0) == target.bool()).float().mean()),
        samples=target.numel(), inactive=int(target.sum()))


def run(steps: int, seed: int, device: str, num_customers: int = 96,
        evaluate_test: bool = False) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    history = generate_history(seed, num_customers)
    splits = ['train', 'val'] + (['test'] if evaluate_test else [])
    batches = {split: prepare_split(history, split, seed) for split in splits}
    channels, context_size = 32, 32
    row_encoder = torch.nn.Linear(8, channels).to(device)
    encoder = RelGTTokenEncoder(channels, 3, pe_layers=2).to(device)
    # Memory indexes customer seeds, all available before the training origin.
    model = RelGT(channels, 1, num_customers, num_layers=2, heads=4,
                  num_centroids=16, global_channels=16).to(device)
    modules = torch.nn.ModuleList([row_encoder, encoder, model])
    optimizer = torch.optim.Adam(modules.parameters(), lr=0.001)
    noise = {
        split: fixed_noise(split, seed, num_customers, context_size).to(device)
        for split in splits
    }

    def predict(split, pe_noise):
        batch = batches[split]
        metadata = {
            key: batch[key].to(device)
            for key in ['node_type', 'hop', 'age', 'edge_index']
        }
        tokens = encoder(row_encoder(batch['x'].to(device)), **metadata,
                         pe_noise=pe_noise)
        return model(tokens, batch['seed_id'].to(device)).flatten()

    @torch.no_grad()
    def evaluate(split):
        modules.eval()
        return metrics(
            predict(split, noise[split]).cpu(), batches[split]['target'])

    initial = evaluate('train')
    started = time.perf_counter()
    best, best_state, best_step = None, None, None
    for step in range(1, steps + 1):
        modules.train()
        optimizer.zero_grad()
        logits = predict('train', torch.randn_like(noise['train']))
        loss = F.binary_cross_entropy_with_logits(
            logits, batches['train']['target'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            validation = evaluate('val')
            if best is None or validation['bce'] < best['bce']:
                best, best_step = validation, step
                best_state = copy.deepcopy(modules.state_dict())
    modules.load_state_dict(best_state)
    prevalence = batches['train']['target'].mean().clamp(1e-6, 1 - 1e-6)
    constant = torch.logit(prevalence).expand(num_customers)
    result = dict(
        kind='simulated_future_inactivity', real_data=False, seed=seed,
        device=device, steps=steps, selected_step=best_step,
        selection='validation BCE only', prediction_origins_days=ORIGINS,
        horizon_days=HORIZON, customers=num_customers,
        accounts=history.account_customer.numel(),
        transactions=history.transaction_day.numel(), initial_train=initial,
        selected_train=evaluate('train'), validation=best,
        constant_baseline_validation=metrics(constant,
                                             batches['val']['target']),
        test_evaluated=evaluate_test, seconds=time.perf_counter() - started,
        parameters=sum(p.numel() for p in modules.parameters()),
        torch_version=torch.__version__)
    if evaluate_test:
        result['test'] = evaluate('test')
        result['constant_baseline_test'] = metrics(constant,
                                                   batches['test']['target'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--customers', type=int, default=96)
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    output = json.dumps(
        run(args.steps, args.seed, args.device, args.customers,
            args.evaluate_test), indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n')
