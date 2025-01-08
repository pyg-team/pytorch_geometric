# Examples for LLM and GNN co-training

| Example                                                          | Description                                                                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [`benchmark_model_archs_rag.py`](./benchmark_model_archs_rag.py) | Script for running a GNN/LLM benchmark on GRetriever while grid searching relevent architecture parameters and datasets. |
| [`minimal_demo.py`](./minimal_demo.py)                           | Minimal demo for WebQSP dataset comparing GNN+LLM vs LLM                                                                 |

NOTE: Evaluating performance on GRetriever with smaller sample sizes may result in subpar performance. It is not unusual for the fine-tuned model/LLM to perform worse than an untrained LLM on very small sample sizes.
