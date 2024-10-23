# Examples for Co-training LLMs and GNNs

| Example                              | Description                                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py) | Example for Retrieval-Augmented Generation (RAG) w/ GNN+LLM by co-training `LLAMA2` with `GAT` for answering questions based on knowledge graph information |
| [`hotpot_qa.py`](./hotpot_qa.py)     | Example for converting conventional Retrieval-Augmented Generation (RAG) into GraphRAG, and how to approximate the recall@5 of a subgraph retrieval method  |
