# Examples for LLM and GNN co-training

| Example                                                  | Description                                                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| [`multihop_download.sh`](./multihop_download.sh)         | Downloads all the components of the multihop dataset.                                                                                      |
| [`multihop_preprocess.py`](./multihop_preprocess.py)     | Preprocesses the dataset to pair questions/answers with components in the knowledge graph. Contains documentation to describe the process. |
| [`rag_generate_multihop.py`](./rag_generate_multihop.py) | Utilizes the sample remote backend in [`g_retriever_utils`](../g_retriever_utils/) to generate subgraphs for the multihop dataset.         |

NOTE: Performance of GRetriever on this dataset has not been evaluated.
