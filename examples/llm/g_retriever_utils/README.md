# Examples for LLM and GNN co-training

| Example                                                          | Description                                                                                                                                                               |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`rag_feature_store.py`](./rag_feature_store.py)                 | A Proof of Concept Implementation of a RAG enabled FeatureStore that can serve as a starting point for implementing a custom RAG Remote Backend                           |
| [`rag_graph_store.py`](./rag_graph_store.py)                     | A Proof of Concept Implementation of a RAG enabled GraphStore that can serve as a starting point for implementing a custom RAG Remote Backend                             |
| [`rag_backend_utils.py`](./rag_backend_utils.py)                 | Utility functions used for loading a series of Knowledge Graph Triplets into the Remote Backend defined by a FeatureStore and GraphStore                                  |
| [`rag_generate.py`](./rag_generate.py)                           | Script for generating a unique set of subgraphs from the WebQSP dataset using a custom defined retrieval algorithm (defaults to the FeatureStore and GraphStore provided) |
| [`benchmark_model_archs_rag.py`](./benchmark_model_archs_rag.py) | Script for running a GNN/LLM benchmark on GRetriever while grid searching relevant architecture parameters and datasets.                                                  |

NOTE: Evaluating performance on GRetriever with smaller sample sizes may result in subpar performance. It is not unusual for the fine-tuned model/LLM to perform worse than an untrained LLM on very small sample sizes.
