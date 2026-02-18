# TXT2QA Deployment Summary

`txt2qa.py` generates synthetic multi-hop question-answer pairs from text documents using a
LangGraph pipeline with artifact extraction, hybrid validation, hard negative mining, and
LLM-as-a-judge evaluation. It supports a local vLLM backend and a cloud NVIDIA NIM backend.

## Quick Start

### vLLM (local GPU)

```bash
python3 examples/llm/txt2qa.py \
  --config examples/llm/txt2qa_config/text_config_vllm.yaml \
  2>&1 | tee examples/llm/logs/vllm_success.log
```

### NVIDIA NIM (API)

```bash
export NVIDIA_API_KEY="your-key-here"
python3 examples/llm/txt2qa.py \
  --config examples/llm/txt2qa_config/text_config_nim.yaml \
  2>&1 | tee examples/llm/logs/nim_success.log
```

## Critical Config Flags

| Flag                    | Default | Description                                      |
| ----------------------- | ------- | ------------------------------------------------ |
| `backend`               | —       | `"vllm"` or `"nim"`                              |
| `gen_model`             | —       | Generation model name (HF or NIM path)           |
| `embedding_model`       | —       | Embedding model name (HF or NIM path)            |
| `num_pairs`             | 3       | QA pairs to generate per input file              |
| `quality_threshold`     | 7.0     | Minimum LLM-judge score to keep a pair (1–10)    |
| `hard`                  | true    | Enable hard mode (complex, multi-step questions) |
| `min_hops` / `max_hops` | 2 / 4   | Reasoning hop range for multi-hop questions      |

## Output

Generated QA pairs are written to `{output_dir}/all_qa_pairs_batch_0.jsonl` as newline-delimited
JSON objects, each containing the question, answer, hop contexts, hard negatives, and quality
scores. See `txt2qa_config/` for full YAML reference and `logs/` for example run output.
