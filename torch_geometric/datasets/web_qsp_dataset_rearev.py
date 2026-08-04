import os
import shutil
from typing import Optional

import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.llm.models import SentenceTransformer


class WebQSPDatasetReaRev(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        limit: Optional[int] = None,
        force_reload: bool = False,
    ):
        self.split = split
        self.model_name = model_name
        self.limit = limit
        self.max_seq_len = 50

        if force_reload and os.path.exists(root):
            shutil.rmtree(root)

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def processed_file_names(self):
        return [f"rearev_{self.split}.pt"]

    def process(self):
        from datasets import load_dataset

        dataset = load_dataset("rmanluo/RoG-webqsp", split=self.split)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SentenceTransformer(self.model_name).to(device)

        data_list = []
        for i, sample in enumerate(
                tqdm(dataset, desc=f"Processing {self.split}")):
            if self.limit and i >= self.limit:
                break
            data = self._process_sample(sample, encoder, device)
            if data is not None:
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _process_sample(self, sample, encoder, device):
        question, answers = sample["question"], [
            a.lower() for a in sample.get("answer", [])
        ]
        triplets = sample["graph"]

        nodes, node_map = [], {}
        src, dst, rels = [], [], []

        def get_idx(text):
            if text not in node_map:
                node_map[text] = len(nodes)
                nodes.append(text)
            return node_map[text]

        for h, r, t in triplets:
            u, v = get_idx(h), get_idx(t)
            src.extend([u, v])
            dst.extend([v, u])
            rels.extend([r, f"inv_{r}"])

        if not nodes:
            return None

        with torch.no_grad():
            x = encoder.encode(nodes).cpu()
            edge_attr = encoder.encode(rels).cpu()

            tok = encoder.tokenizer([question], return_tensors='pt',
                                    padding=True, truncation=True,
                                    max_length=encoder.max_seq_length)
            tok = {k: v.to(device) for k, v in tok.items()}
            out = encoder.model(**tok)

            raw_seq, raw_mask = out.last_hidden_state[0].cpu(
            ), tok["attention_mask"][0].cpu()
            curr_len = raw_seq.size(0)

            if curr_len > self.max_seq_len:
                q_tokens, q_mask = raw_seq[:self.
                                           max_seq_len], raw_mask[:self.
                                                                  max_seq_len]
            else:
                pad = self.max_seq_len - curr_len
                q_tokens = torch.cat([raw_seq, torch.zeros(pad, 384)])
                q_mask = torch.cat([raw_mask, torch.zeros(pad)])

        y = torch.tensor([1.0 if n.lower() in answers else 0.0 for n in nodes])

        seed_mask = torch.tensor([
            1.0 if len(n) > 2 and n.lower() in question.lower() else 0.0
            for n in nodes
        ])

        return Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long),
                    edge_attr=edge_attr,
                    edge_type=torch.zeros(len(src), dtype=torch.long), y=y,
                    seed_mask=seed_mask, question_tokens=q_tokens.unsqueeze(0),
                    question_mask=q_mask.unsqueeze(0), node_text=nodes,
                    edge_text=rels, question_text=question,
                    answer_text=answers)
