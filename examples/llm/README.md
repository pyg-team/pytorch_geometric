# Examples for Co-training LLMs and GNNs

| Example                                      | Description                                                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py)         | Example for Retrieval-Augmented Generation (RAG) w/ GNN+LLM by co-training `LLAMA2` with `GAT` for answering questions based on knowledge graph information |
| [`g_retriever_utils/`](./g_retriever_utils/) | Contains multiple scripts for benchmarking GRetriever's architecture and evaluating different retrieval methods.                                            |

| [`nvtx_examples/`](./nvtx_examples/)         | Contains examples of how to wrap functions using the NVTX profiler for CUDA runtime analysis.                                                                                                  |
| [`molecule_gpt.py`](./molecule_gpt.py)       | Example for MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction                                                                                         |
| [`glem.py`](./glem.py)                       | Example for [GLEM](https://arxiv.org/abs/2210.14709), a GNN+LLM co-training model via variational Expectation-Maximization (EM) framework on node classification tasks to achieve SOTA results |
| [`git_mol.py`](./git_mol.py)                 | Example for GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text                                                                                      |
| [`hotpot_qa.py`](./hotpot_qa.py)             | Example for converting adapting the retrieval step of conventional Retrieval-Augmented Generation (RAG) for use with G-retriever, and how to approximate the precision/recall of a subgraph retrieval method. Uses the HotPotQA dataset as an example since it is multihop in nature. |
