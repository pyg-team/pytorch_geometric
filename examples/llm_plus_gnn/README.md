# Examples for LLM and GNN co-training

| Example                              | Description                                                                                                                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py) | Example for Retrieval-Augmented Generation (RAG) w/ GNN+LLM by co-training LLAMA2 with GAT for answering questions based on knowledge graph information                                        |
| [`glem.py`](./glem.py)               | Example for [GLEM](https://arxiv.org/abs/2210.14709), a GNN+LLM co-training model via variational Expectation-Maximization (EM) framework on node classification tasks to achieve SOTA results |

## Run GLEM for getting SOTA result on ogbn-products dataset

`python glem.py --train_with_ext_pred`
