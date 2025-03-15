# Link Prediction Examples

## Attract-Repel Link Prediction

The [`attract_repel.py`](attract_repel.py) example demonstrates how to use the `AttractRepel` wrapper for Graph Neural Networks to improve link prediction performance.

### Usage

```bash
# Run with GCN on Cora
python attract_repel.py --dataset=Cora --model=GCN

# Try GraphSAGE on CiteSeer
python attract_repel.py --dataset=CiteSeer --model=GraphSAGE

# Experiment with different attract/repel ratios
python attract_repel.py --dataset=PubMed --model=GAT --attract_ratio=0.7
```

### Performance

The Attract-Repel approach typically shows significant improvements over traditional embeddings:

| Dataset | Traditional (AUC) | Attract-Repel (AUC) | Improvement | R-fraction |
|---------|-------------------|---------------------|-------------|------------|
| Cora    | 0.6624            | 0.8945              | +0.2321     | 0.4075     |
| PubMed  | 0.8977            | 0.9607              | +0.0630     | 0.5062     |
| CiteSeer| 0.8067            | 0.8206              | +0.0139     | 0.4869     |

### How It Works

The `AttractRepel` wrapper splits node embeddings into two components:

1. **Attract component**: Similar nodes in this space are more likely to connect
2. **Repel component**: Similar nodes in this space are less likely to connect

This approach is based on the paper "Pseudo-Euclidean Attract-Repel Embeddings for Undirected Graphs" (Peysakhovich et al., 2023).