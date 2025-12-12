"""
This example uses the subgraphs from the validation set of the WebQSP dataset,
feeds the subgraph to a trained ReaRev model, builds the reasoning paths
and uses a LLM to answer the sample question from the validation set.
"""
import argparse
from typing import Any, Dict, List

import networkx as nx
import torch
from torch_geometric.datasets import WebQSPDatasetReaRev
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from torch_geometric.llm.models import LLM
from torch_geometric.datasets import WebQSPDatasetReaRev
from torch_geometric.nn.models import ReaRev

def top_k_paths(probs: torch.Tensor, batch: Data, top_k: int = 3) -> List[List[Dict[str, Any]]]:
    probs = probs.view(-1).cpu()
    batch_idx, edge_index = batch.batch.cpu(), batch.edge_index.cpu()
    seed_mask = batch.seed_mask.cpu()

    flatten = lambda l: [item for sublist in l for item in sublist] if l and isinstance(l[0], list) else l
    all_node_texts = flatten(batch.node_text)
    edges_txt = flatten(batch.edge_text)

    # We'll make the graph undirected, and not worry about inverse relations for verabalization
    g = nx.Graph()
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    for i, (u, v) in enumerate(edges):
        rel = edges_txt[i] if i < len(edges_txt) else ""

        if g.has_edge(u, v):
            existing_idx = g[u][v].get('idx')
            existing_rel = (
                edges_txt[existing_idx]
                if existing_idx is not None and existing_idx < len(edges_txt)
                else ""
            )
            # Display non-inverse edges for verbalization
            if existing_rel.startswith("inv_") and not rel.startswith("inv_"):
                g[u][v]['idx'] = i
        else:
            g.add_edge(u, v, idx=i)

    def get_fallback_seed(graph_nodes, question_text):
        q_words = set(question_text.lower().split())
        best_node, max_overlap = None, 0
        for n_idx in graph_nodes:
            if n_idx < len(all_node_texts):
                node_str = all_node_texts[n_idx].lower()
                overlap = len(q_words.intersection(node_str.split()))
                if overlap > max_overlap:
                    max_overlap, best_node = overlap, n_idx
        return best_node

    paths_data = []
    for i in range(batch.num_graphs):
        mask = batch_idx == i
        nodes = mask.nonzero().view(-1)
        if nodes.numel() == 0:
            paths_data.append([])
            continue

        local_probs = probs[nodes]
        k = min(top_k, local_probs.numel())
        scores, local_top_idx = torch.topk(local_probs, k)
        
        seeds = nodes[seed_mask[nodes] == 1.0].tolist()
        if not seeds:
            q_text = batch.question_text[i]
            fallback = get_fallback_seed(nodes.tolist(), q_text)
            if fallback is not None:
                seeds = [fallback]

        graph_paths = []
        for score, idx in zip(scores, local_top_idx):
            target = nodes[idx].item()
            best_path, min_len = None, float('inf')

            for seed in seeds:
                if seed == target: 
                    if best_path is None: best_path = [seed]
                    continue
                try:
                    path = nx.shortest_path(g, seed, target)
                    if len(path) < min_len:
                        best_path, min_len = path, len(path)
                except nx.NetworkXNoPath: continue
            
            if best_path and len(best_path) > 1:
                edge_idxs = []
                for u, v in zip(best_path, best_path[1:]):
                    edge_data = g.get_edge_data(u, v)
                    if isinstance(edge_data, dict) and 'idx' in edge_data:
                         edge_idxs.append(edge_data['idx'])
                    else:
                         edge_idxs.append(list(edge_data.values())[0]['idx'])
                graph_paths.append({"score": score.item(), "nodes": best_path, "edges": edge_idxs})
            else:
                graph_paths.append({"score": score.item(), "nodes": [target], "edges": []})
        
        paths_data.append(graph_paths)

    return paths_data

def verbalize_paths(batch: Data, raw_paths: List[List[Dict]]) -> List[str]:
    contexts = []
    
    flatten = lambda l: [item for sublist in l for item in sublist] if l and isinstance(l[0], list) else l
    nodes_txt = flatten(batch.node_text)
    edges_txt = flatten(batch.edge_text)

    for paths in raw_paths:
        lines = []
        for p in paths:
            nodes, edges, score = p['nodes'], p['edges'], p['score']
            
            if any(n >= len(nodes_txt) for n in nodes): continue

            if not edges:
                lines.append(f"Entity: {nodes_txt[nodes[0]]}")
                continue

            parts = []
            for i, edge_idx in enumerate(edges):
                rel = edges_txt[edge_idx] if edge_idx < len(edges_txt) else "unknown"
                parts.append(f"{nodes_txt[nodes[i]]} --[{rel}]-->")
            
            parts.append(nodes_txt[nodes[-1]])
            lines.append(f"{''.join(parts)}")

        contexts.append("\n".join(lines) if lines else "No relevant paths found.")

    return contexts

@torch.no_grad()
def inference_step(model, llm, batch, top_k=3):
    model.eval()
    
    probs = model(
        batch.question_tokens, batch.question_mask, batch.x, 
        batch.edge_index, batch.edge_type, batch.edge_attr, 
        batch.seed_mask, batch.batch
    )
    raw_paths = top_k_paths(probs, batch, top_k)
    path_ctx = verbalize_paths(batch, raw_paths)

    prompt_template = """Based on the reasoning paths, please answer the given question. Please
    keep the answer as simple as possible and return all the possible answers
    as a list.

    Reasoning Paths: {context} 
    Question: {question}"""
    
    questions = batch.question_text
    prompts = [
        prompt_template.format(question=q, context=ctx)
        for q, ctx in zip(questions, path_ctx)
    ]
    
    return {
        "questions": questions,
        "context_paths": path_ctx,
        "prompts": prompts,
        "predicted_answers": llm.inference(prompts),
        "ground_truth": batch.answer_text
    }


def display_results(res, title: str):
    print(f"\n{title}")
    print("\n" + "="*80)
    for q, context, truth, pred in zip(
        res["questions"], res["context_paths"], res["ground_truth"], res["predicted_answers"]
    ):
        truth_str = ", ".join(truth) if isinstance(truth, list) else truth

        pred_clean = pred.split("Reasoning Paths:")[0].strip()
        if "Answer:" in pred_clean:
            pred_clean = pred_clean.split("Answer:")[-1].strip()

        print(f"Q: {q}")
        print("-" * 20)

        print("Context:")
        for line in context.split("\n"):
            if line.strip():
                print(f"  • {line}")

        print("-" * 20)
        print(f"Truth: {truth_str}")
        print(f"Pred : {pred_clean}")
        print("=" * 80 + "\n")



def main(root, model_path, inference_limit, llm_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use examples from the validation set for inference
    val_dataset = WebQSPDatasetReaRev(root=root, split="validation", limit=inference_limit)
    val_loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load trained model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    model = ReaRev(**config).to(device)
    model.load_state_dict(state_dict)
    
    sys_prompt = (
    "You are an expert assistant that answers questions. "
    "Just give the answer, without explanation."
    )
    llm = LLM(model_name=llm_model_name, sys_prompt=sys_prompt).eval()

    # Run inference on examples from the validation set
    for i, example in enumerate(val_loader):
        if i >= inference_limit:
            break
        example = example.to(device)
        res = inference_step(model, llm, example, top_k=3)
        display_results(res, f"Inference {i+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ReaRev on WebQSP.")
    parser.add_argument("--root", type=str, default="./data/webqsp", help="Dataset root directory.")
    parser.add_argument("--inference_limit", type=int, default=10, help="Number of examples to load.")
    parser.add_argument("--model_path", type=str, default="../outputs/rearev/rearev.pth", help="Path to the trained model.")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="LLM model name.")
    args = parser.parse_args()

    main(
        root=args.root,
        inference_limit=args.inference_limit,
        model_path=args.model_path,
        llm_model_name=args.llm_model_name,
    )