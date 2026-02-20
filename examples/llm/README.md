# Examples for Co-training LLMs and GNNs

| Example                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py)   | Example helper functions for using the [G-retriever](https://arxiv.org/abs/2402.07630) GNN+LLM module in PyG. Includes an [example repo](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example) for [Neo4j](https://neo4j.com) integration with an associated [blog post](https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/) demonstrating 2x accuracy gains over LLMs on real medical data. For a complete end-to-end pipeline (KG Creation, Subgraph Retrieval, GNN+LLM Finetuning, Testing, LLM Judge Eval), see [`txt2kg_rag.py`](./txt2kg_rag.py). For a native PyG implementation without external graph databases, see [gretriever-stark-prime](https://github.com/puririshi98/gretriever-stark-prime/tree/main).      |
| [`molecule_gpt.py`](./molecule_gpt.py) | Example for MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction. Supports MoleculeGPT and InstructMol dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [`glem.py`](./glem.py)                 | Example for [GLEM](https://arxiv.org/abs/2210.14709), a GNN+LLM co-training model via variational Expectation-Maximization (EM) framework on node classification tasks to achieve SOTA results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [`git_mol.py`](./git_mol.py)           | Example for GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [`protein_mpnn.py`](./protein_mpnn.py) | Example for [Robust deep learning--based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [`txt2kg_rag.py`](./txt2kg_rag.py)     | Full end 2 end RAG pipeline using TXT2KG and Vector and Graph RAG with a GNN to achieve state of the art results. Uses the [techQA dataset](https://paperswithcode.com/dataset/techqa) but can be extended to handle any RAG dataset with a corpus of documents and an associated set of Q+A pairs to be split for train/eval/test. See [Stanford GNN+LLM Talk](https://www.nvidia.com/en-us/on-demand/session/other25-nv-0003/) for more details. Note that the TechQA data requires only a single document to answer each question so it can be viewed as a toy example. To see significant accuracy boosts from GNN+LLM TXT2KG based RAG, use data that requires multiple text chunks to answer a question. In cases where single document can answer, basic RAG should be sufficient. |
| [`txt2qa.py`](./txt2qa.py)             | Synthetic multi-hop QA generation pipeline from text documents. Supports vLLM (local GPU) and NVIDIA NIM (API) backends. See [`DEPLOYMENT_SUMMARY.md`](./DEPLOYMENT_SUMMARY.md) for run instructions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## TXT2QA Quick Start

### Running

**vLLM (local GPU):**

```bash
python3 examples/llm/txt2qa.py \
  --config examples/llm/txt2qa_config/text_config_vllm.yaml
```

**NVIDIA NIM (API):**

```bash
export NVIDIA_API_KEY="your-key-here"
python3 examples/llm/txt2qa.py \
  --config examples/llm/txt2qa_config/text_config_nim.yaml
```

Output is written to `{output_dir}/all_qa_pairs_batch_0.jsonl`. See [`DEPLOYMENT_SUMMARY.md`](./DEPLOYMENT_SUMMARY.md) for config flags and details.

### Building Containers for TXT2QA

TXT2QA requires both PyG and vLLM. You can start from either base container:

#### Option A: Starting from NGC vLLM Container

```bash
# Inside the vLLM container, install PyG:
pip install torch_geometric[full,rag]
# Or, for a development install from a local clone:
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric && pip install .[full,rag]
```

#### Option B: Starting from NGC PyG Container

```bash
# Install vLLM pre-built wheel (CUDA 13.0 example):
pip install "https://github.com/vllm-project/vllm/releases/download/v0.15.0/vllm-0.15.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

For other CUDA versions or installation methods, see the [vLLM installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-wheels).

> **Note:** The vLLM wheel may pull in a `flash-attn` build that is incompatible with the
> existing environment. If you hit import errors related to flash-attention, run:
>
> ```bash
> pip uninstall flash-attn flash_attn -y
> ```
>
> This resolves the issue — vLLM will fall back to its built-in attention backends.
