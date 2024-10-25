# Examples for Co-training LLMs and GNNs

| Example                              | Description                                                                                                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py) | Example for Retrieval-Augmented Generation (RAG) w/ GNN+LLM by co-training `LLAMA2` with `GAT` for answering questions based on knowledge graph information                                           |
| [`hotpot_qa.py`](./hotpot_qa.py)     | Example for converting adapting the retrieval step of conventional Retrieval-Augmented Generation (RAG) for use with G-retriever, and how to approximate the precision/recall of a subgraph retrieval method. |
