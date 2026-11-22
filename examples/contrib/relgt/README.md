# RelGT: relational prediction

This example accompanies `RelGTTokenEncoder` and `RelGT` in
`torch_geometric.contrib.nn.models`. It implements the existing
[Relational Graph Transformer](https://arxiv.org/abs/2505.10960) method using
PyG primitives. Adapted code follows
[the released implementation](https://github.com/snap-stanford/relgt/tree/19e423ca3e7cac761130aba790857f2dc3a46ef7);
the original MIT notice is retained beside the library model.

## Simulated future customer inactivity

From a PyG checkout with PyTorch and PyG's core dependencies installed:

```bash
python -m examples.contrib.relgt.synthetic --steps 120 --device cpu
```

The generator creates 240 days of transactions for 96 customers, each owning
one or two accounts. Customers can become dormant and resume activity through
a stochastic process. The target is **no transaction in the next 30 days**;
it is computed from future events, not from customer IDs or a feature copied
into the label. Latent simulation states are never supplied to the model.

Prediction origins are days 90, 150 and 210 for training, validation and test.
Every customer is included regardless of future activity. Input graphs and
recent-activity aggregates use only transactions strictly before each origin.
Training labels finish before validation begins. Normalization uses declared
fixed units, not statistics from future splits. Global memory contains only
customer seeds that existed before the training origin.

The example trains the row projection, five token components, local/global
attention and prediction head. Validation BCE selects a checkpoint; the
output reports ROC-AUC, BCE, accuracy and a training-prevalence baseline.
The optional `--evaluate-test` flag evaluates the selected checkpoint once
on the later simulated test window. Structural noise is fresh in training
and fixed by customer and prediction time in evaluation.

These are chronological held-out **simulation** results, not evidence about
real banking customers. The simulator omits many real-world causes of
inactivity; it demonstrates temporal data construction and model usage.
No database, RelBench or TorchFrame is required. Tiny deterministic fixtures
in the tests independently check whether the complete training path overfits.

## Interface

```python
from torch_geometric.contrib.nn.models import RelGT, RelGTTokenEncoder

encoder = RelGTTokenEncoder(channels=512, num_node_types=num_tables)
model = RelGT(channels=512, out_channels=1, num_nodes=num_entities,
              num_layers=4, heads=4, num_centroids=4096,
              global_channels=256, dropout=0.3)
tokens = encoder(x=row_embeddings, node_type=node_type, hop=hop,
                 age=age_in_days, edge_index=context_edges,
                 pe_noise=structural_noise)
scores = model(tokens, seed_id=global_entity_ids)
```

Rows are `[B,K,512]`, type/hop/age metadata `[B,K]`, and noise `[B,K,1]`.
Edges index flattened token occurrences, with no edges between contexts.
The seed occupies token zero. IDs consistently map into `[0,num_entities)`;
cache identity additionally includes task row, data revision, split and time.
Outputs retain `[B,C]`. Use external BCE-with-logits, regression or multiclass
losses. Batch normalization requires `B>=2` for training; evaluation supports
one sample. `mode='local'` removes global attention.

## Sampling and memory semantics

`data.py` provides CPU temporal sampling, eligible fallback rows, duplicate
occurrence edges, deterministic cache validation, and collation. It rebuilds
adjacency/fingerprints per request and targets small inspectable fixtures.
Large datasets require a pre-indexed adapter rather than this inspectable
example sampler. Unknown timestamps require an explicit caller assertion that
a row is safe to treat as static.
The caller must handle dated edges and historical feature versions.

Memory contains representative embeddings (centroids), EMA sums/counts,
normalization, assignments and population counts, all in `state_dict`.
Training reads memory before updating it; evaluation is immutable. Duplicate
seed IDs use the last occurrence for assignment while every occurrence enters
EMA statistics. `reset_memory()` starts a new memory history while retaining
learned weights.

Initial populations count all supplied entity IDs, matching the reference.
Real temporal experiments must define an entity universe available at the
training cutoff. Filtering local neighbors alone does not establish historical
global-memory validity. For a chronological evaluation, fit weights, preprocessing and memory using
data available at the training cutoff. Freeze them before scoring later
samples. Historical replay at each training timestamp needs a separate
training protocol.

The example uses float32 on one device. Early EMA centroids can have large
magnitudes. Mixed precision and synchronization of memory across devices
require separate validation.

## Using a RelBench task

In RelBench's Formula 1 `driver-top3` task, the prediction row identifies a
driver and a cutoff date. The same driver can appear at several cutoffs.
The following adapter turns one row of a pinned RelBench split into the
input for `sample_context`. It expects a dictionary mapping the primary keys
in the drivers table to global row positions in your graph. Supply revision
identifiers for the downloaded database and task tables when calling it.

```python
from examples.contrib.relgt.data import PredictionSample


def driver_sample(task, split, position, driver_index, db_revision,
                  task_revision):
    row = task.get_table(split).df.iloc[position]
    return PredictionSample(
        dataset_revision=db_revision,
        task_revision=task_revision,
        split=split,
        task_row_id=str(position),
        seed_index=driver_index[int(row[task.entity_col])],
        prediction_time=int(row[task.time_col].value // 1_000_000_000),
    )
```

With RelBench installed, load the task via
`relbench.tasks.get_task("rel-f1", "driver-top3", download=True)`.
Assign split positions before sorting or filtering. Set `RelGTGraph.time` to
the first time each record is available, in Unix seconds. The sampler includes
rows available at the cutoff. For a task requiring strictly earlier events,
filter events at the cutoff before sampling. Mark a row as static only when
its features and relationships were known throughout the prediction period.

Project each table's eligible attributes to the same embedding width, using
training-period statistics. Use `collate_contexts` to index these embeddings
and offset the edges for each sampled context. Supply the resulting tensors
to the encoder above. Extract the task's target column separately for the
loss. For large databases, pre-index row eligibility and neighbors to avoid
scanning the complete graph for every prediction.
