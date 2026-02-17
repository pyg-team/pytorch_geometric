# Examples for Co-training LLMs and GNNs

| Example                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`g_retriever.py`](./g_retriever.py)   | Example helper functions for using the [G-retriever](https://arxiv.org/abs/2402.07630) GNN+LLM module in PyG. Includes an [example repo](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example) for [Neo4j](https://neo4j.com) integration with an associated [blog post](https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/) demonstrating 2x accuracy gains over LLMs on real medical data. For a complete end-to-end pipeline (KG Creation, Subgraph Retrieval, GNN+LLM Finetuning, Testing, LLM Judge Eval), see [`txt2kg_rag.py`](./txt2kg_rag.py). For a native PyG implementation without external graph databases, see [gretriever-stark-prime](https://github.com/puririshi98/gretriever-stark-prime/tree/main).      |
| [`nvtx_examples/`](./nvtx_examples/)   | Contains examples of how to wrap functions using the NVTX profiler for CUDA runtime analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [`molecule_gpt.py`](./molecule_gpt.py) | Example for MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction. Supports MoleculeGPT and InstructMol dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [`glem.py`](./glem.py)                 | Example for [GLEM](https://arxiv.org/abs/2210.14709), a GNN+LLM co-training model via variational Expectation-Maximization (EM) framework on node classification tasks to achieve SOTA results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [`git_mol.py`](./git_mol.py)           | Example for GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [`protein_mpnn.py`](./protein_mpnn.py) | Example for [Robust deep learning--based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [`txt2kg_rag.py`](./txt2kg_rag.py)     | Full end 2 end RAG pipeline using TXT2KG and Vector and Graph RAG with a GNN to achieve state of the art results. Uses the [techQA dataset](https://paperswithcode.com/dataset/techqa) but can be extended to handle any RAG dataset with a corpus of documents and an associated set of Q+A pairs to be split for train/eval/test. See [Stanford GNN+LLM Talk](https://www.nvidia.com/en-us/on-demand/session/other25-nv-0003/) for more details. Note that the TechQA data requires only a single document to answer each question so it can be viewed as a toy example. To see significant accuracy boosts from GNN+LLM TXT2KG based RAG, use data that requires multiple text chunks to answer a question. In cases where single document can answer, basic RAG should be sufficient. |
| [`txt2qa.py`](./txt2qa.py)             | Synthetic QA data generation pipeline for creating high-quality, multi-hop reasoning question-answer pairs from text documents. Supports both local vLLM models and NVIDIA NIM API. Features include artifact extraction, validation, hard negative mining, and LLM-based evaluation. Perfect for generating training data for RAG systems. See [detailed guide](#txt2qa-synthetic-qa-generation) below.                                                                                                                                                                                                                                                                                                                                                                                      |

---

## TXT2QA: Synthetic QA Generation

The `txt2qa.py` script generates high-quality, multi-hop reasoning question-answer pairs from text documents, suitable for training RAG systems and fine-tuning language models.

### Features

- **Dual Backend Support**: Use local models via vLLM or cloud models via NVIDIA NIM API
- **Multi-hop Reasoning**: Generate complex questions requiring synthesis of multiple facts
- **Artifact Extraction**: Automatically extract semantic artifacts (concepts, definitions, relationships)
- **Validation Pipeline**: Hybrid validation using embedding similarity and substring matching
- **Hard Negative Mining**: Automatically mine challenging but plausible wrong answers
- **LLM-based Evaluation**: Quality assessment using LLM-as-a-judge
- **Configurable Complexity**: Control question types, reasoning patterns, and difficulty

### Quick Start

#### Option 1: Local Models (vLLM)

**Requirements:**
- NVIDIA GPU with sufficient VRAM (recommend 24GB+ for 7B models)
- CUDA-compatible drivers

**Setup:**
```bash
# Install dependencies
pip install vllm torch faiss-gpu

# Run with default configuration
python examples/llm/txt2qa.py --config examples/llm/txt2qa_config/text_config_vllm.yaml
```

**Configuration** (`text_config_vllm.yaml`):
```yaml
backend: "vllm"
gen_model: "Qwen/Qwen2.5-7B-Instruct"
embedding_model: "BAAI/bge-base-en-v1.5"
vllm_gpu_memory_utilization:
  generation: 0.5   # 50% GPU memory for generation
  embedding: 0.3    # 30% GPU memory for embeddings
```

**Expected Output:**
- Quality scores: 8-9/10
- Throughput: ~1-2 QA pairs/minute
- GPU memory usage: ~17GB peak

See full log: [`logs/vllm_success.log`](logs/vllm_success.log)

#### Option 2: NVIDIA NIM API

**Requirements:**
- NVIDIA API key (get one at https://build.nvidia.com/)
- No GPU required!

**Setup:**
```bash
# Set API key
export NVIDIA_API_KEY="your-api-key-here"

# Run with NIM configuration
python examples/llm/txt2qa.py --config examples/llm/txt2qa_config/text_config_nim.yaml
```

**Configuration** (`text_config_nim.yaml`):
```yaml
backend: "nim"
gen_model: "nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1"
embedding_model: "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2"
```

**Expected Output:**
- Quality scores: 9-10/10 (higher quality than local models)
- Throughput: ~0.5-1 QA pairs/minute (API latency)
- Cost: Pay-per-use API pricing

See full log: [`logs/nim_success.log`](logs/nim_success.log)

### Pipeline Stages

The generation pipeline consists of 8 stages:

1. **Loading**: Read and segment input documents
2. **Artifact Extraction**: Extract key concepts, definitions, relationships
3. **QA Generation**: Generate initial question-answer pairs
4. **Validation**: Verify answers exist in source text using hybrid approach
5. **Hard Negative Mining**: Find plausible but incorrect answer segments
6. **Negative Generation**: Create convincing wrong answers
7. **Evaluation**: Score quality using LLM-as-a-judge
8. **Filtering**: Keep only high-quality pairs above threshold

### Configuration Options

Key parameters in config files:

```yaml
# Processing
num_pairs: 3                    # QA pairs per document
num_negatives: 1                # Wrong answers per question
dedup_threshold: 0.6            # Similarity threshold for validation
quality_threshold: 7.0          # Minimum quality score (1-10)

# Question Complexity
hard: true                      # Enable hard mode (complex questions)
min_complexity: 4               # Minimum complexity level (1-5)
min_hops: 2                     # Minimum reasoning hops
max_hops: 4                     # Maximum reasoning hops

# Query Types Distribution
query_type_distribution:
  multi_hop: 0.4                # Questions requiring multiple facts
  structural: 0.3               # Questions about document structure
  contextual: 0.3               # Questions about context/relationships

# Reasoning Types Distribution
reasoning_type_distribution:
  factual: 0.15                 # Direct fact retrieval
  causal: 0.15                  # Cause-effect relationships
  relational: 0.15              # Entity relationships
  inferential: 0.15             # Inference required
  temporal: 0.15                # Time-based reasoning
  procedural: 0.15              # Step-by-step processes
  visual: 0.1                   # Visual/spatial reasoning
```

### Output Format

Generated QA pairs are saved in JSONL format with rich metadata:

```json
{
  "question": "How does X relate to Y in the context of Z?",
  "answer": "Detailed answer with context...",
  "query_type": "multi_hop",
  "reasoning_type": "causal",
  "hop_count": 3,
  "hop_contexts": [...],
  "context_segments": [...],
  "hard_negative_segment_ids": [5, 12, 18],
  "negative_answers": ["Wrong answer 1", "Wrong answer 2"],
  "quality_score": 9.5,
  "evaluation": {
    "relevance": {"score": 10, "justification": "..."},
    "accuracy": {"score": 9, "justification": "..."},
    "context_support": {"score": 10, "justification": "..."},
    "clarity": {"score": 9, "justification": "..."}
  }
}
```

### Performance Comparison

| Metric | vLLM (Local) | NIM (API) |
|--------|-------------|-----------|
| Quality Score | 8.5/10 | 9.3/10 |
| Throughput | 1-2 pairs/min | 0.5-1 pairs/min |
| GPU Required | Yes (24GB+) | No |
| Setup Complexity | Medium | Easy |
| Cost | Hardware only | Pay-per-use |
| Model Size | 7B params | 253B params |
| Best For | High throughput | Highest quality |

### Troubleshooting

**Issue: "AttributeError: 'LLMClient' object has no attribute 'is_local'"**
- **Fixed in current version** - Update to latest code

**Issue: "AssertionError in FAISS search"**
- **Cause**: Embedding dimension mismatch from old indexes
- **Solution**: Delete old indexes: `rm -rf output_dir/embedding_data/`

**Issue: "Expecting value: line 1 column 1 (char 0)"**
- **Fixed in current version** - Better JSON parsing with markdown cleanup

**Issue: All QA pairs filtered out (0 validated)**
- **Cause**: FAISS L2 distance vs cosine similarity bug
- **Fixed in current version** - Proper distance conversion

**Issue: Out of GPU memory**
- **Solution**: Reduce `vllm_gpu_memory_utilization` in config
- **Alternative**: Use NIM API (no GPU required)

### Advanced Usage

**Custom Data Sources:**
```yaml
input_dir: "/path/to/your/documents"
output_dir: "/path/to/output"

# Optional: Download from HuggingFace
huggingface_repo_id: "your-org/your-dataset"
huggingface_repo_type: "dataset"
huggingface_files:
  corpus: "corpus.zip"
```

**Model Selection (vLLM):**
```yaml
models:
  qwen_1.5b: "Qwen/Qwen2.5-1.5B-Instruct"    # Faster, lower quality
  qwen_7b: "Qwen/Qwen2.5-7B-Instruct"        # Balanced
  llama_8b: "meta-llama/Llama-3.1-8B"        # Alternative
```

**Model Selection (NIM):**
```yaml
models:
  mixtral: "nvdev/mistralai/mixtral-8x22b-instruct-v0.1"
  llama_70b: "nvdev/meta/llama-3.1-70b-instruct"
  llama_405b: "nvdev/meta/llama-3.1-405b-instruct"          # Highest quality
  nemotron: "nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1"  # Best overall
```

### Example Outputs

**Sample High-Quality Question (Multi-hop, 4 reasoning steps):**
> "How does the CVSS Base Score of 4.2 for CVE-2018-2800 relate to the CVSS Base Score of 7.4 for CVE-2018-2783, and what does this imply for the overall risk assessment of these vulnerabilities in the context of the IBM Security SiteProtector System?"

**Quality Assessment:**
- Relevance: 10/10
- Accuracy: 9/10
- Context Support: 10/10
- Clarity: 9/10
- **Overall: 9.5/10**

### Citation

If you use this tool in your research, please cite:

```bibtex
@misc{pytorch_geometric_txt2qa,
  title={TXT2QA: Synthetic Multi-hop QA Generation for RAG Systems},
  author={PyTorch Geometric Team},
  year={2024},
  howpublished={https://github.com/pyg-team/pytorch_geometric}
}
```

### Support

- **Documentation**: See this README and config file comments
- **Example Logs**: Check `logs/` directory for success examples
- **Issues**: Report bugs at https://github.com/pyg-team/pytorch_geometric/issues
- **Discussions**: Ask questions at https://github.com/pyg-team/pytorch_geometric/discussions
