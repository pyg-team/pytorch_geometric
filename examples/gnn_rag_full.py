"""
This example builds a full KG from the WebQSP val dataset
then matches node entities to the full KG, builds a subgraph
and feeds the subgraph to a trained ReaRev model to get the reasoning paths
"""
import argparse
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Set

import networkx as nx
import torch
from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.datasets import WebQSPDatasetReaRev
from torch_geometric.data import Data
from torch_geometric.nn.models import ReaRev
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

def build_full_kg(datasets):
    print("Building full KG...")
    e2id, r2id = {}, {}
    n_txt, n_embs = [], []
    r_embs, r_txts = [], []
    r_keys, r_keys_txt = {}, {}
    edges = [[], [], [], []] 
    seen = set()

    def _get(n, emb):
        k = n.strip().lower()
        if k not in e2id:
            e2id[k] = len(e2id)
            n_txt.append(k)
            n_embs.append(emb.cpu())
        return e2id[k]

    for ds in datasets:
        for s in tqdm(ds, desc="Merging"):
            if not (hasattr(s, 'edge_index') and s.edge_index.numel() > 0): continue
            
            l2g = {i: _get(n, s.x[i]) for i, n in enumerate(s.node_text)}
            edge_iter = s.edge_index.t().tolist()
            
            for idx, (src, dst) in enumerate(edge_iter):
                g_src, g_dst = l2g[src], l2g[dst]
                rv = s.edge_attr[idx].cpu()
                rt_raw = s.edge_text[idx] if hasattr(s, "edge_text") else None
                
                rid = None
                if isinstance(rt_raw, str) and rt_raw.strip():
                    k = rt_raw.strip().lower()
                    rid = r_keys_txt.setdefault(k, len(r_embs))
                    if rid == len(r_embs):
                        r2id[k] = rid
                        r_embs.append(rv)
                        r_txts.append(rt_raw)
                else:
                    k = rv.numpy().tobytes()
                    rid = r_keys.setdefault(k, len(r_embs))
                    if rid == len(r_embs):
                        r_embs.append(rv)
                        r_txts.append(f"rel_{rid}")
                        r2id[f"rel_{rid}"] = rid
                
                if (key := (g_src, g_dst, rid)) in seen: continue
                seen.add(key)
                
                edges[0].append(g_src)
                edges[1].append(g_dst)
                edges[3].append(rid)
                etype = int(s.edge_type[idx].item()) if hasattr(s, "edge_type") else 0
                edges[2].append(etype)

    full = Data(
        x=torch.stack(n_embs) if n_embs else torch.empty((0, 384)),
        edge_index=torch.tensor(edges[:2], dtype=torch.long),
        edge_attr=torch.stack(r_embs)[edges[3]] if r_embs else torch.empty((0, 384)),
        edge_type=torch.tensor(edges[2]),
        edge_rel_ids=torch.tensor(edges[3])
    )
    full.node_text = n_txt
    full.rel_text = r_txts or [f"rel_{i}" for i in range(len(r_embs))]
    return full, e2id, r2id

def build_subgraph(full_kg, anchors, depth=2, max_nodes=1000, max_edges=10000):
    if not anchors: return Data(), {}
    
    sub, idx, _, mask = k_hop_subgraph(anchors, depth, full_kg.edge_index, 
                                       relabel_nodes=True, num_nodes=full_kg.num_nodes)
    
    if sub.numel() > max_nodes:
        sub = sub[:max_nodes]
        keep = (idx[0] < max_nodes) & (idx[1] < max_nodes)
        idx = idx[:, keep]
        if mask.dtype == torch.bool:
            mask = mask.nonzero().view(-1)
        mask = mask[keep]
    else:
        if mask.dtype == torch.bool:
            mask = mask.nonzero().view(-1)

    if idx.size(1) > max_edges:
        keep_edges = torch.arange(max_edges, device=idx.device)
        idx = idx[:, keep_edges]
        mask = mask[keep_edges]
            
    data = Data(
        x=full_kg.x[sub],
        edge_index=idx,
        edge_attr=full_kg.edge_attr[mask],
        edge_type=full_kg.edge_type[mask],
        edge_rel_ids=full_kg.edge_rel_ids[mask]
    )
    data.node_text = [full_kg.node_text[i] for i in sub.tolist()]
    data.node_global_ids = sub
    if hasattr(full_kg, "rel_text"):
        data.edge_text = [full_kg.rel_text[i] for i in data.edge_rel_ids.tolist()]
    
    g2l = {g.item(): i for i, g in enumerate(sub)}
    data.seed_mask = torch.zeros(sub.size(0))
    valid_anchors = [g2l[g] for g in anchors if g in g2l]
    data.seed_mask[valid_anchors] = 1.0
    
    return data

def find_anchor_from_entity(entity: str, e2id: Dict[str, int]) -> Optional[int]:
    """
    Resolve the node id for a given entity string.
    Tries exact match first, then a simple substring fallback.
    """
    key = entity.strip().lower()
    if key in e2id:
        return e2id[key]
    
    for name, idx in e2id.items():
        if key in name:
            return idx
    return None

def encode_q(q, enc, max_len=50, device=None):
    device = device or torch.device("cpu")
    tokens = enc.tokenizer([q], return_tensors="pt", padding=True, truncation=True, 
                           max_length=min(max_len, enc.max_seq_length))
    inputs = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad(): 
        out = enc.model(**inputs)
        
    rt, rm = out.last_hidden_state[0], inputs["attention_mask"][0]
    
    if (pad := max_len - rt.size(0)) > 0:
        rt = torch.cat([rt, torch.zeros(pad, rt.size(1), device=device)])
        rm = torch.cat([rm, torch.zeros(pad, device=device, dtype=rm.dtype)])
        
    return rt[:max_len].unsqueeze(0).cpu(), rm[:max_len].unsqueeze(0).cpu()

def top_k_paths(probs: torch.Tensor, batch: Data, top_k: int = 3) -> List[List[Dict[str, Any]]]:
    probs = probs.view(-1).cpu()
    batch_idx = batch.batch.cpu() if hasattr(batch, 'batch') else torch.zeros(probs.size(0), dtype=torch.long)
    edge_index = batch.edge_index.cpu()
    seed_mask = batch.seed_mask.cpu()

    flatten = lambda l: [item for sublist in l for item in sublist] if l and isinstance(l[0], list) else l
    all_node_texts = flatten(batch.node_text)
    
    if hasattr(batch, 'edge_text') and batch.edge_text is not None:
         edges_txt = flatten(batch.edge_text)
    else:
        edges_txt = [""] * edge_index.size(1)

    # We'll make the graph undirected
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
    num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
    
    for i in range(num_graphs):
        mask = batch_idx == i
        nodes = mask.nonzero().view(-1)
        if nodes.numel() == 0:
            paths_data.append([])
            continue

        local_probs = probs[nodes]
        k = min(top_k, local_probs.numel())
        scores, local_top_idx = torch.topk(local_probs, k)
        
        seeds = nodes[seed_mask[nodes] == 1.0].tolist()
        q_text = batch.question_text[i] if isinstance(batch.question_text, list) else batch.question_text
        
        if not seeds:
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
    
    if hasattr(batch, 'edge_text') and batch.edge_text is not None:
        edges_txt = flatten(batch.edge_text)
    else:
        edges_txt = [] 

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
                if edge_idx < len(edges_txt):
                    rel = edges_txt[edge_idx]
                else:
                    rel = "related_to" # Fallback if text missing
                parts.append(f"{nodes_txt[nodes[i]]} --[{rel}]-->")
            
            parts.append(nodes_txt[nodes[-1]])
            lines.append(f"{''.join(parts)}")

        contexts.append("\n".join(lines) if lines else "No relevant paths found.")

    return contexts

def main(root, model_path, question, entity, max_nodes, max_edges):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WebQSPDatasetReaRev(root=root, split="validation")
    q_str = question.strip()
    
    # Build full KG
    full_kg, e2id, _ = build_full_kg([ds])
    print(f"KG: {full_kg.num_nodes} nodes, {full_kg.num_edges} edges.")

    # Find anchor from input entity
    anchor = find_anchor_from_entity(entity, e2id)
    if anchor is None:
        print(f"Entity not found in KG: {entity}")
        return
    anchors = [anchor]
    anchor_txt = full_kg.node_text[anchor] if anchor < len(full_kg.node_text) else entity
    print(f"Using anchor entity: {anchor_txt}")

    # Load Model
    cp = torch.load(model_path, map_location=dev)
    model = ReaRev(**cp['config']).to(dev)
    model.load_state_dict(cp['state_dict'])
    model.eval()
    

    # build subgraph from anchor
    sub = build_subgraph(full_kg, anchors, max_nodes=max_nodes, max_edges=max_edges).to(dev)
    if sub.num_nodes == 0:
        print("Empty subgraph generated.")
        return
    sub.batch = torch.zeros(sub.num_nodes, dtype=torch.long, device=dev)
    
    # Encode input question
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(dev)
    q_tokens, q_mask = encode_q(q_str, enc, device=dev)
    sub.question_tokens = q_tokens.to(dev)
    sub.question_mask = q_mask.to(dev)
        
    probs = model(sub.question_tokens, sub.question_mask, sub.x, sub.edge_index, 
                  sub.edge_type, sub.edge_attr, sub.seed_mask, sub.batch)
    
    sub.question_text = [q_str]
    raw_paths = top_k_paths(probs, sub, 3)
    ctx_str = verbalize_paths(sub, raw_paths)[0]

    print(ctx_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/webqsp")
    parser.add_argument("--model_path", default="../outputs/rearev/rearev.pth")
    parser.add_argument("--question", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Maximum nodes to keep in the subgraph.")
    parser.add_argument("--max-edges", type=int, default=10000,
                        help="Maximum edges to keep in the subgraph.")
    args = parser.parse_args()
    main(args.root, args.model_path, args.question, args.entity, args.max_nodes, args.max_edges)
    