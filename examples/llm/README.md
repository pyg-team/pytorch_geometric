# Examples for Co-training LLMs and GNNs

| Example                                      | Description                                                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py)         | Example for Retrieval-Augmented Generation (RAG) w/ GNN+LLM by co-training `LLAMA2` with `GAT` for answering questions based on knowledge graph information |
| [`g_retriever_utils/`](./g_retriever_utils/) | Contains multiple scripts for benchmarking GRetriever's architecture and evaluating different retrieval methods.                                            |
| [`multihop_rag/`](./multihop_rag/)           | Contains starter code and an example run for building a Multi-hop dataset using WikiHop5M and 2WikiMultiHopQA                                               |
| [`nvtx_examples/`](./nvtx_examples/)         | Contains examples of how to wrap functions using the NVTX profiler for CUDA runtime analysis.                                                               |
