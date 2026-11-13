r"""Public benchmark for :obj:`kumo_gretriever.py`: business questions over
RelBench ``rel-amazon`` that ask for the top customers or books among those
that will churn.

A question is anchored at a date ``T`` and asks for the ``k`` best customers
(``user`` task) or books (``item`` task) under a ranking rule whose
eligibility is a *future* label: no review in ``(T, T + 91 days]`` (the
RelBench ``user-churn``/``item-churn`` definition).  Families: ``RV1``
(churning customers by trailing 91-day spend ``prior_ltv``, optionally within
a segment), ``RV3`` (churning customers by spend per offer cost under
other/Gold/Platinum tiers, funded greedily within a budget, which the drawn
budgets never make bind before ``k``), ``P2`` (books
going dormant by trailing review revenue, optionally within a genre or price
band) and ``P5`` (books going dormant by ``factor * price - prior_ltv``).

Questions are drawn with a fixed mix: customers 71% ``RV1`` (62% without a
segment) and 29% ``RV3`` (64% default offer policy, 18% explicit offer costs,
18% Gold tier only); books 72% ``P2`` and 28% ``P5``.  The candidate universe
is every entity reviewed in the trailing 91 days at the anchor (no
subsampling).  Candidates carry only pre-anchor features (``prior_ltv``,
``prior_reviews``, ``prior_avg_rating``; books also ``price``, ``genre`` and
their category set).  A pool holds 64 candidates: the gold entities and
distractors matched to them on the standardized feature vector (same
``prior_ltv`` cents when available, nearest neighbours otherwise).  Pools are
therefore *label-dependent*: the candidate set is built around the realized
answer, and the task is to rank it, not to retrieve it from the database.  The
candidates are listed in an order that depends only on their ids, the task and
the anchor, so the order carries no label.  Ranking them by the question's rule
as if all of them churned (``kumo_gretriever.py --arm rule``) shows what the
forecast adds.

The in-context examples are 256 labeled rows at the latest fit date whose
91-day window closes before the anchor (2012-07-05 for every anchor from
2012-10-04 on, so test questions condition on 2012 labels), sampled with
``num_neighbors=(16, 16)``; relational tables are cut at the anchor, so no
later event reaches the model. Strict recall of an answer is ``|ids & gold| /
k`` when exactly ``k`` distinct pool ids are returned, else ``0``. No Amazon
data is redistributed; RelBench downloads ``rel-amazon`` on demand, at the Hub
revision :obj:`RELBENCH_REVISION`.
"""
import hashlib
import json
import math
import random
from decimal import ROUND_HALF_EVEN, Decimal
from html import unescape
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MARKER = '<|relemb|>'
# The pinned revision of `stanford-star/relbench-v1` on the Hugging Face Hub:
RELBENCH_REVISION = '21d7284df2a9621c71f21bc4a1a30b7e8effb8dd'
HORIZON = pd.Timedelta(days=91)
POOL_SIZE = 64
CONTEXT_ROWS = 256
NUM_NEIGHBORS = (16, 16)
FEATURES = ('prior_ltv', 'prior_reviews', 'prior_avg_rating')
# The 32 rel-amazon anchors, 91 days apart: 19 fit dates (labels for the
# in-context examples), the default training anchor, then the 12 test anchors.
ANCHORS = tuple(
    str(t.date()) for t in pd.date_range('2008-01-10', periods=32, freq='91D'))
FIT_DATES, TRAIN_ANCHOR = ANCHORS[:19], ANCHORS[19]
KS = (3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
RV1_KS = KS + (4, 6, 7, 9, 14, 18, 22, 35, 45)  # Unsegmented RV1 questions.
SEGMENT_KS, RV3_KS = (5, 10, 20), (6, 12, 20, 25)
BUDGETS = (10000, 25000, 50000, 100000, 250000, 500000)
FACTORS = ('0.2', '0.35', '0.5')
# RV3 offer policy: Gold/Platinum tiers at the 75th/90th percentile of
# trailing spend; costs for other/Gold/Platinum, plus the overrides some
# questions spell out (explicit costs are rendered as floats).
TIERS, COSTS = ('36.79', '69.95'), ('4', '10', '25')
COST_OVERRIDES = ((4, 40, 400), (5, 15, 40), (4, 20, 60))
GOLD_SCOPE = (None, 'prior_ltv', 'ge_lt', [36.79, 69.95])
GENRES = ('Literature & Fiction', 'Mystery, Thriller & Suspense',
          'Christian Books & Bibles', "Children's Books", 'Romance', 'History')
TEMPLATES = {
    'RV1': (
        "As of {T}, pull the {k} highest-value customers we're about to lose "
        'from our books store and list them so we can send a retention offer '
        'before they walk.',
        'Show me, in order of prior lifetime value, the {k} book customers '
        'who are set to churn as of {T}. These are the accounts I want to hit '
        'with a loyalty incentive.',
    ),
    'RV3': (
        "It's {T} and the CFO capped our win-back spend at {budget}. Which "
        'realized churners give us the most protected revenue per dollar of '
        'offer cost? Fund them greedily by that ratio until the budget is '
        'exhausted and list who makes the cut.',
        'Help me spend {budget} wisely this quarter ({T}). Among our lapsing '
        'book readers who are confirmed to churn, which top {k} give the '
        'highest prior-LTV-to-tier-cost return and still fit under budget? '
        "That's the retention list I need.",
    ),
    'P2': (
        'As of {T}, pull the {k} highest-earning book titles that are about '
        'to go dormant next quarter (no reviews coming) so we can drop a '
        'promotion on them before they stall.',
        'Rank the {k} highest prior-revenue books that our model says will be '
        'dormant (zero reviews) in the quarter following {T}. I want to run '
        'promotions on the valuable ones going cold, not the ones already '
        'dead weight.',
    ),
    'P5': (
        'Which {k} books should we consider cutting loose as of {T}? Look '
        'only at dormant titles and rank them by carrying cost minus trailing '
        'revenue, most costly on top.',
        'Give me a {T} delist candidate list: the {k} dormant books whose '
        'carrying cost most exceeds their prior revenue, ranked from worst '
        'offender down.',
    ),
}
# Question scopes: (bracket label, field, op, value).
USER_SCOPES = (
    ('Among churning customers with trailing LTV under $50', 'prior_ltv', 'lt',
     50.0),
    ('Among churning customers with trailing LTV between $50 and $200',
     'prior_ltv', 'ge_le', [50.0, 200.0]),
    ('Among one-review customers (exactly 1 prior review)', 'prior_reviews',
     'eq', 1.0),
    ('Among light reviewers (2 to 4 prior reviews)', 'prior_reviews', 'ge_le',
     [2.0, 4.0]),
    ('Among engaged reviewers (5 or more prior reviews)', 'prior_reviews',
     'ge', 5.0),
    ('Among customers whose prior average rating is under 4 stars',
     'prior_avg_rating', 'lt', 4.0),
    ('Among customers whose prior average rating is a perfect 5',
     'prior_avg_rating', 'eq', 5.0),
)
ITEM_SCOPES = (
    ('Among books under $10', 'price', 'lt', 10.0),
    ('Among books $10-25', 'price', 'ge_lt', [10.0, 25.0]),
    ('Among premium books over $25', 'price', 'ge', 25.0),
) + tuple(
    (f'Within the {g} genre', 'genre', 'contains', g.lower()) for g in GENRES)
# Bin boundaries spelled out in the mid/premium price questions.
PRICE_NOTES = {'ge_lt': '$10.00 <= price < $25.00', 'ge': 'price >= $25.00'}
OPS = {
    'lt': lambda v, x: v < x,
    'ge': lambda v, x: v >= x,
    'eq': lambda v, x: v == x,
    'ge_le': lambda v, x: (v >= x[0]) & (v <= x[1]),
    'ge_lt': lambda v, x: (v >= x[0]) & (v < x[1]),
}
# The relational schema of `sdm` (structured-data-models): table ->
# {column: semantic type}; time columns cut every table at the anchor.
STYPES = dict(
    customer=dict(customer_id='id', first_seen='datetime'),
    product=dict(product_id='id', price='numerical', brand='categorical',
                 category_json='categorical', first_seen='datetime'),
    review=dict(review_time='datetime', customer_id='id', product_id='id',
                rating='numerical', verified='categorical'),
)
TIME_COLUMNS = dict(customer='first_seen', product='first_seen',
                    review='review_time')
ENTITY = dict(user=('customer', 'customer_id'), item=('product', 'product_id'))
RELATIONSHIPS = [
    dict(left_table='review', left_column=c, right_table=t, right_column=c)
    for t, c in ENTITY.values()
]


def dec(value) -> Decimal:
    return Decimal(str(value))


def allowed(candidate: Dict[str, Any], constraint) -> bool:
    r"""Whether a candidate satisfies the question's declared constraint."""
    c = constraint or {}
    if not c.get('field'):
        return True
    if c['field'] == 'genre':
        return c['value'] in candidate.get('categories', ())
    value, x = candidate['features'].get(c['field']), c['value']
    if value is None:
        return False
    x = [dec(b) for b in x] if isinstance(x, list) else dec(x)
    return bool(OPS[c['op']](dec(value), x))


def family_eligible(candidate: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    r"""The P5 rule that a book must have a positive price."""
    price = candidate['features'].get('price')
    return policy['family'] != 'P5' or (price is not None and dec(price) > 0)


def eligible(candidate: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    r"""The question's constraint plus :func:`family_eligible`."""
    return family_eligible(candidate, policy) and allowed(
        candidate, policy['constraint'])


def offer_cost(features: Dict[str, Any], policy: Dict[str, Any]) -> Decimal:
    value, costs = dec(
        features['prior_ltv']), [dec(c) for c in policy['costs']]
    gold, platinum = (dec(t) for t in policy['tiers'])
    if value >= platinum:
        return costs[2]
    return costs[1] if value >= gold else costs[0]


def score(features: Dict[str, Any], policy: Dict[str, Any]) -> Decimal:
    r"""Ranking score of an eligible candidate under the family's rule."""
    value = dec(features['prior_ltv'])
    if policy['family'] == 'P5':
        return dec(policy['carrying_cost_factor']) * dec(
            features['price']) - value
    if policy['family'] == 'RV3':
        ratio = value / offer_cost(features, policy)
        return ratio.quantize(Decimal('0.000000001'), rounding=ROUND_HALF_EVEN)
    return value


def sort_key(candidate: Dict[str, Any], policy: Dict[str, Any]):
    r"""Score descending, trailing spend descending, numeric id ascending."""
    f = candidate['features']
    return -score(f, policy), -dec(f['prior_ltv']), int(candidate['id'])


def gold_ids(candidates: List[Dict], labels: Dict[str, int],
             policy: Dict[str, Any]) -> List[str]:
    r"""Eligible churners in rank order, after the RV3 budget cut."""
    chosen = [
        c for c in candidates if labels[c['id']] == 1 and eligible(c, policy)
    ]
    chosen.sort(key=lambda c: sort_key(c, policy))
    funded, spent = [], Decimal(0)
    for c in chosen:
        if policy['family'] == 'RV3':
            spent += offer_cost(c['features'], policy)
            if spent > dec(policy['budget']):
                break
        funded.append(c['id'])
    return funded[:policy['requested_count']]


def strict_recall(ids, gold, pool, k: int) -> float:
    r"""``|ids & gold| / k`` for exactly ``k`` distinct pool ids, else 0."""
    ids = list(ids)
    if len(ids) != k or len(set(ids)) != k or not set(ids) <= set(pool):
        return 0.0
    return len(set(ids) & set(gold)) / k


def make_policy(family: str, k: int, constraint=None, budget=None, factor=None,
                costs=COSTS) -> Dict[str, Any]:
    policy = dict(family=family, requested_count=int(k),
                  constraint=dict(constraint
                                  or {}), ranking='historical_value')
    if family == 'RV3':  # The budget is spelled as a float in the rules.
        policy.update(ranking='historical_value_per_offer_cost',
                      budget=str(float(budget)), tiers=list(TIERS),
                      costs=list(costs))
    if family == 'P5':
        policy.update(ranking='carrying_cost_minus_historical_value',
                      carrying_cost_factor=str(factor))
    return policy


def neutral_order(task: str, anchor: str, ids: List[str]) -> List[str]:
    r"""Candidate ids sorted by the SHA-256 hash of task, anchor date and id,
    then by numeric id: an order that depends only on the pool, the task and
    the anchor, never on labels.
    """
    def key(i: str):
        text = f'kumo-neutral-order-v1|{task}|{anchor[:10]}|{i}'
        return hashlib.sha256(text.encode()).hexdigest(), int(i)

    return sorted(ids, key=key)


def render_context(candidates: List[Dict]) -> str:
    r"""One JSON row per candidate followed by the relational marker."""
    rows = []
    for c in candidates:
        row = {'features': dict(sorted(c['features'].items())), 'id': c['id']}
        if 'categories' in c:
            row['categories'] = c['categories']
        text = json.dumps(row, separators=(',', ':'), ensure_ascii=False)
        rows.append(f'{text} {MARKER}')
    return '\n'.join(rows)


def render_question(text: str, policy: Dict[str, Any]) -> str:
    r"""The question followed by the shared task rules."""
    lines = [text, '', 'Task rules:']
    if policy['family'] == 'RV3':
        gold, platinum = policy['tiers']
        other, gold_cost, platinum_cost = policy['costs']
        lines.append(
            'Offer tiers use the displayed historical value: Platinum at or '
            f'above ${platinum}; Gold from ${gold} inclusive to ${platinum} '
            f'exclusive; other below ${gold}. Costs are ${platinum_cost}, '
            f'${gold_cost}, and ${other}, respectively. '
            f'Budget: ${policy["budget"]}.')
        lines.append(  # Fixed wording, typo included, as the reported
            # checkpoints were trained on it:
            'Rank by historical value divided by offer cost, rounded '
            'to9decimalplaces. Fund the sorted prefix whose cumulative cost '
            'stays within budget. Return its first requested number of IDs.')
    lines.append('Rank ties by historical value descending, then numeric '
                 "candidate ID ascending. Use the original question's "
                 'constraints and ranking rule.')
    lines.append(f'Return exactly {policy["requested_count"]} candidate IDs '
                 'in best-first order, as comma-separated values, without '
                 'additional text. If there are fewer eligible candidates, '
                 'report that explicitly instead of inventing IDs.')
    return '\n'.join(lines)


def _genre(cats) -> str:
    cats = [c for c in (unescape(str(x)) for x in cats) if c and c != 'Books']
    return ' > '.join(cats[:2]) if cats else 'none'


def _population_gold(frame: pd.DataFrame, policy: Dict[str, Any]) -> List[str]:
    r"""Top-``k`` answer over all live entities (seeds the pool)."""
    c, df = policy['constraint'], frame[frame['churn'] == 1]
    if c.get('field') == 'genre':
        df = df[[c['value'] in s for s in df['categories']]]
    elif c.get('field'):
        df = df[OPS[c['op']](df[c['field']], c['value'])]
    if policy['family'] == 'P5':
        df = df[df['price'] > 0]
    ltv, cost = df['prior_ltv'], None
    if policy['family'] == 'P5':
        s = float(policy['carrying_cost_factor']) * df['price'] - ltv
    elif policy['family'] == 'RV3':
        tiers = [float(t) for t in policy['tiers']]
        costs = [float(c) for c in policy['costs']]
        cost = np.where(ltv >= tiers[1], costs[2],
                        np.where(ltv >= tiers[0], costs[1], costs[0]))
        s = (ltv / cost).round(9)
    else:
        s = ltv
    df = df.assign(score=s, idnum=df.index.to_numpy(), cost=cost)
    df = df.sort_values(['score', 'prior_ltv', 'idnum'],
                        ascending=[False, False, True])
    if policy['family'] == 'RV3':
        df = df[df['cost'].cumsum() <= float(policy['budget'])]
    return [str(i) for i in df.index[:policy['requested_count']]]


def _distractors(frame: pd.DataFrame, gold: List[str], n: int,
                 seed: int) -> List[str]:
    r"""Non-gold entities at a gold's ``prior_ltv`` cents (nearest in
    standardized reviews/rating), then nearest in the standardized feature
    space; seeded tie-break.
    """
    if n <= 0:
        return []
    rng = np.random.RandomState(seed)
    ids, x = frame.index.to_numpy(), frame[list(FEATURES)].to_numpy(float)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    z = (x - x.mean(axis=0)) / std
    gold_pos = frame.index.get_indexer([int(g) for g in gold])
    cents = np.rint(frame['prior_ltv'].to_numpy() * 100).astype(np.int64)
    by_cents = pd.Series(np.arange(len(ids))).groupby(cents).indices
    taken = np.zeros(len(ids), dtype=bool)
    taken[gold_pos] = True
    chosen, per = [], max(1, math.ceil(n / len(gold)))
    for g in sorted(gold_pos, key=lambda p: ids[p]):
        cands = [p for p in by_cents.get(cents[g], ()) if not taken[p]]
        if not cands:
            continue
        d = np.linalg.norm(z[cands, 1:] - z[g, 1:], axis=1)
        order = np.lexsort((rng.permutation(len(cands)), d))[:per]
        got = [cands[i] for i in order]
        taken[got] = True
        chosen += got
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        dist = np.full(len(ids), np.inf)
        for g in gold_pos:
            dist = np.minimum(dist, ((z - z[g])**2).mean(axis=1))
        free = np.flatnonzero(~taken)
        free = free[rng.permutation(free.size)]
        chosen += list(free[np.argsort(dist[free], kind='stable')])
    return [str(ids[p]) for p in chosen[:n]]


def _rv3_clause(costs, k: int, gold_only: bool) -> str:
    r"""Parenthetical of RV3 questions that spell out their offer policy."""
    gold, platinum = TIERS
    other, gold_cost, platinum_cost = (f'{float(c):g}' for c in costs)
    default = [dec(c) for c in costs] == [dec(c) for c in COSTS]
    text = ' ('
    if gold_only:
        text += ('Platinum accounts are excluded (handled by key-account '
                 'managers), so fund Gold-tier churners only, i.e. realized '
                 'churners whose prior lifetime value is at or above '
                 f'${gold} and below ${platinum}; ')
    text += ('value tiers are defined on prior lifetime value in dollars and '
             'cents across the full customer base: Platinum = prior lifetime '
             f'value at or above ${platinum} (the 90th percentile), Gold = at '
             f'or above ${gold} (the 75th percentile) and below ${platinum}, '
             f'other = below ${gold}; ')
    text += ('playbook offer costs apply: ' if default else
             'for this scenario override the playbook offer costs to ')
    return text + (f'${platinum_cost} per Platinum customer, ${gold_cost} per '
                   f'Gold customer, ${other} otherwise; list the first {k} '
                   'funded customers in ratio order)')


def _draw(rng: random.Random, task: str, anchor: str):
    r"""A random (question text, policy) pair of the task's families."""
    u = rng.random()
    if task == 'user':
        family = 'RV1' if u < 0.71 else 'RV3'
    else:
        family = 'P2' if u < 0.72 else 'P5'
    scope = budget = factor = clause = None
    costs = COSTS
    if family == 'RV1':
        scope = None if rng.random() < 0.62 else rng.choice(USER_SCOPES)
        k = rng.choice(RV1_KS if scope is None else SEGMENT_KS)
    elif family == 'RV3':
        v = rng.random()
        if v < 0.64:  # Default offer policy and 12 ids.
            k, budget = 12, rng.choice(BUDGETS)
        else:  # Explicit costs, or the Gold tier at the default costs.
            k, scope = rng.choice(RV3_KS), None if v < 0.82 else GOLD_SCOPE
            budget = 100000 if scope is None else 50000
            costs = rng.choice(COST_OVERRIDES) if scope is None else COSTS
            costs = tuple(str(float(c)) for c in costs)
            clause = _rv3_clause(costs, k, scope is not None)
    else:
        if family == 'P2':
            scope = rng.choice((None, ) + ITEM_SCOPES)
        k = rng.choice(KS)
        factor = rng.choice(FACTORS) if family == 'P5' else None
    constraint = dict(zip(
        ('field', 'op', 'value'), scope[1:])) if scope else {}
    policy = make_policy(family, k, constraint, budget, factor, costs)
    text = rng.choice(TEMPLATES[family]).format(
        T=anchor, k=k, budget='' if budget is None else f'${budget:,}')
    if scope is not None and scope[0] is not None:
        text = f'[{scope[0]}] {text}'
        note = PRICE_NOTES.get(scope[2]) if scope[1] == 'price' else None
        if note is not None:
            text += ('\nConstraint clarification from the declared dataset '
                     f'price bin: {note}.')
    if factor is not None:
        text += (f' (use a carrying-cost factor of {factor}x the list price '
                 'when scoring)')
    return text + (clause or ''), policy


def _candidate(task: str, eid: str, row: pd.Series) -> Dict[str, Any]:
    f = {k: float(row[k]) for k in FEATURES}
    c = {'id': eid, 'features': f}
    if task == 'item':
        f['price'] = None if np.isnan(row['price']) else row['price']
        f['genre'] = row['genre']
        c['categories'] = list(row['categories'])
    return c


class _Amazon:
    r"""rel-amazon tables plus per-anchor feature frames and `sdm` samplers."""
    def __init__(self, root: str):
        import pyarrow.parquet as pq
        from relbench import load_dataset
        db = load_dataset(root, revision=RELBENCH_REVISION).db_dir
        columns = dict(
            review=[
                'review_time', 'customer_id', 'product_id', 'rating',
                'verified'
            ], customer=['customer_id'],
            product=['product_id', 'price', 'brand', 'category'])
        self.review, self.customer, product = (pq.read_table(
            db / f'{n}.parquet', columns=c).to_pandas()
                                               for n, c in columns.items())
        cats = [
            None if c is None else [str(x) for x in c]
            for c in product.pop('category')
        ]
        product['category_json'] = [
            None if c is None else json.dumps(
                c, ensure_ascii=False, separators=(',', ':')) for c in cats
        ]
        product['genre'] = [_genre(c or []) for c in cats]
        product['categories'] = [
            sorted({unescape(x).lower()
                    for x in (c or [])}) for c in cats
        ]
        product['price'] = [
            round(float(p), 2) if math.isfinite(p) else np.nan
            for p in product['price']
        ]
        for frame, (_, col) in ((self.customer, ENTITY['user']),
                                (product, ENTITY['item'])):
            first = self.review.groupby(col)['review_time'].min()
            frame['first_seen'] = frame[col].map(first)
        self.product = product.set_index('product_id')
        self.review['price'] = self.review['product_id'].map(
            self.product['price'])
        self._frames, self._sampler = {}, {}

    def universe(self, task: str, anchor: str) -> pd.DataFrame:
        r"""Entities reviewed in ``(T - 91d, T]``: prior features and the
        ``(T, T + 91d]`` churn label, indexed by integer id.
        """
        if (task, anchor) not in self._frames:
            col, t = ENTITY[task][1], pd.Timestamp(anchor)
            time = self.review['review_time']
            prior = self.review[(time > t - HORIZON) & (time <= t)]
            future = self.review.loc[(time > t) & (time <= t + HORIZON), col]
            g = prior.groupby(col).agg(ltv=('price', 'sum'),
                                       n=('rating', 'size'),
                                       s=('rating', 'sum'))
            frame = pd.DataFrame(index=g.index)
            frame['prior_ltv'] = [round(float(v), 2) for v in g['ltv']]
            frame['prior_reviews'] = g['n'].astype(float)
            frame['prior_avg_rating'] = [
                round(int(s) / int(n), 2) for s, n in zip(g['s'], g['n'])
            ]
            frame['churn'] = (~g.index.isin(future.unique())).astype(int)
            if task == 'item':
                frame = frame.join(
                    self.product[['price', 'genre',
                                  'categories']]).dropna(subset=['genre'])
            self._frames[task, anchor] = frame
        return self._frames[task, anchor]

    def sampler(self, anchor: str):
        r"""`sdm` relational sampler over the tables cut at the anchor."""
        if anchor not in self._sampler:
            import sdm
            self._sampler.clear()
            t, tables = pd.Timestamp(anchor), {}
            frames = dict(customer=self.customer, review=self.review,
                          product=self.product.reset_index())
            for name, stypes in STYPES.items():
                df = frames[name]
                df = df.loc[df[TIME_COLUMNS[name]] <= t, list(stypes)]
                tables[name] = sdm.TableTensor.from_pandas(
                    df.reset_index(drop=True), stypes=stypes)
            data = sdm.RelationalData(tables=tables,
                                      relationships=RELATIONSHIPS)
            self._sampler[anchor] = data.sampler(time_columns=TIME_COLUMNS)
        return self._sampler[anchor]

    def sample(self, sampler, task: str, ids: List[int], time: pd.Timestamp,
               churn=None):
        r"""Sample one task table; returns ``(x, related_tables, y)``."""
        import sdm
        table, col = ENTITY[task]
        frame = pd.DataFrame({col: np.asarray(ids, dtype=np.int64)})
        frame['prediction_time'] = time
        stypes = {col: 'id', 'prediction_time': 'datetime'}
        if churn is not None:
            frame['churn'], stypes['churn'] = churn, 'categorical'
        task_table = sdm.TableTensor.from_pandas(frame, stypes=stypes)
        out = sampler(
            task_table.drop_columns(list(stypes)[2:]),
            task_link=dict(task_column=col, table=table, table_column=col),
            task_time_column='prediction_time', num_neighbors=NUM_NEIGHBORS,
            temporal_strategy='last')
        for name, column in TIME_COLUMNS.items():
            related = out.related_tables.tables.get(name)
            if related is not None:
                times = related.select_columns([column]).to_pandas()[column]
                if (pd.to_datetime(times) > time).any():
                    raise ValueError(f'Post-anchor row sampled from {name}')
        y = task_table['churn'] if churn is not None else None
        return out.task_table, out.related_tables, y

    def context(self, sampler, task: str, anchor: str, seed: int) -> Dict:
        r"""256 labeled fit-period rows whose horizon closes by the anchor."""
        t = pd.Timestamp(anchor)
        date = max(d for d in FIT_DATES if pd.Timestamp(d) + HORIZON <= t)
        frame = self.universe(task, date)
        rows = np.random.RandomState(seed).permutation(len(frame))
        frame = frame.iloc[rows[:CONTEXT_ROWS]]
        if set(frame['churn'].unique()) != {0, 1}:
            raise ValueError('Context must contain both churn classes')
        x, related, y = self.sample(sampler, task, list(frame.index),
                                    pd.Timestamp(date),
                                    frame['churn'].to_numpy())
        return dict(x_context=x, y_context=y, related_context_tables=related)


def build_benchmark(
    root: str = 'stanford-star/relbench-v1/rel-amazon',
    anchors=(TRAIN_ANCHOR, ),
    tasks=('item', 'user'),
    n_questions: int = 32,
    seed: int = 50,
) -> List[Dict[str, Any]]:
    r"""Build ``n_questions`` records per (anchor, task).

    ``root`` is a local ``rel-amazon`` directory or a Hub spec in
    ``stanford-star/relbench-v1`` (such as the default), fetched at
    :obj:`RELBENCH_REVISION`.  Each record holds the prompt (``question``),
    the rendered candidate rows (``context``), the structured
    ``candidates``/``policy``/``labels``/``gold``, and ``pool``
    (``candidate_ids`` plus the `sdm` ``model_input``; the context tensors are
    shared by all records of an (anchor, task)).  The sampled ``model_input``
    is reproducible only with single-threaded Arrow
    (``pyarrow.set_cpu_count(1)``, as ``kumo_gretriever.py`` sets), since a
    multi-threaded hash join reorders the graph's edges.
    """
    source, rng, records = _Amazon(root), random.Random(seed), []
    for anchor in anchors:
        sampler = source.sampler(anchor)
        for task in tasks:
            frame = source.universe(task, anchor)
            context = source.context(sampler, task, anchor, seed)
            made = 0
            for _ in range(20 * n_questions):
                if made == n_questions:
                    break
                text, policy = _draw(rng, task, anchor)
                gold = _population_gold(frame, policy)
                if len(gold) < policy['requested_count']:
                    continue
                ids = gold + _distractors(frame, gold, POOL_SIZE - len(gold),
                                          rng.getrandbits(32))
                ids = neutral_order(task, anchor, ids)
                candidates = [
                    _candidate(task, i, frame.loc[int(i)]) for i in ids
                ]
                labels = {i: int(frame.at[int(i), 'churn']) for i in ids}
                x, related, _ = source.sample(sampler, task,
                                              [int(i) for i in ids],
                                              pd.Timestamp(anchor))
                pool = dict(
                    candidate_ids=ids,
                    model_input=dict(context, x_query=x,
                                     related_query_tables=related))
                records.append(
                    dict(anchor=anchor, task=task, family=policy['family'],
                         k=policy['requested_count'],
                         question=render_question(text, policy),
                         context=render_context(candidates),
                         candidates=candidates, policy=policy, labels=labels,
                         gold=gold_ids(candidates, labels, policy), pool=pool))
                made += 1
    return records
