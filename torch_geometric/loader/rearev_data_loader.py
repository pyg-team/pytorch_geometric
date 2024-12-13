import torch
from torch_geometric.data import Data, DataLoader
from transformers import AutoTokenizer
import json
import os
import numpy as np  # Added to fix undefined name 'np'

class BasicDataLoader:
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        self.batch_size = config['batch_size']
        print("Init called!")
        self.tokenize = tokenize
        self._initialize(config, word2id, relation2id, entity2id)
        print("Loading file")
        self._load_file(config, data_type)
        print("Preparing data")
        self._prepare_data()
        self.build_rel_words(self.tokenize)

    def _initialize(self, config, word2id, relation2id, entity2id):
        print("Initializing")
        self.config = config
        self.word2id, self.relation2id, self.entity2id = word2id, relation2id, entity2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.num_relations = len(relation2id)
        if config.get('use_inverse_relation', False):
            self.num_relations *= 2
        if config.get('use_self_loop', False):
            self.num_relations += 1

    def _load_file(self, config, data_type):
        self.data = []
        file_path = f"{config['data_folder']}{data_type}.json"
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            lines = len(f.readlines())
        with open(file_path, 'r') as f:
            print("Number of lines: ", lines)
            iter = 0
            for line in f:
                line = json.loads(line)
                if (iter % 1000 == 0):
                    print(f"Line {iter} out of {lines}")
                if 'entities' in line:
                    self.data.append(line)
                iter += 1
        print(f"Loaded {len(self.data)} samples.")

    def _prepare_data(self):
        self.graphs = [self._create_graph(sample) for sample in self.data]
        self.data_loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=True)

    def _create_graph(self, sample):
        entity_map = {ent: idx for idx, ent in enumerate(sample.get('entities', []))}
        x = torch.zeros(len(entity_map), self.num_relations)
        edges, edge_attrs = [], []

        for head, rel, tail in sample['subgraph']['tuples']:
            if head in entity_map and tail in entity_map:
                h = entity_map[head]
                t = entity_map[tail]
                r = self.relation2id.get(rel, len(self.relation2id))
                edges.append([h, t])
                edge_attrs.append(r)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def reset_batches(self):
        self.data_loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=True)

    def build_rel_words(self, tokenize):
        # Tokenizes relation surface forms.
        max_rel_words = 0
        rel_words = []
        if 'metaqa' in getattr(self, 'data_file', ''):
            for rel in self.relation2id:
                words = rel.split('_')
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
            # print(rel_words)
        else:
            for rel in self.relation2id:
                rel = rel.strip()
                fields = rel.split('.')
                try:
                    words = fields[-2].split('_') + fields[-1].split('_')
                    max_rel_words = max(len(words), max_rel_words)
                    rel_words.append(words)
                    # print(rel, words)
                except Exception:  # Changed bare except to except Exception
                    words = ['UNK']
                    rel_words.append(words)
                    pass
                # words = fields[-2].split('_') + fields[-1].split('_')

        self.max_rel_words = max_rel_words
        if tokenize == 'lstm':
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words),
                                     len(self.word2id), dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words),
                                         len(self.word2id), dtype=int)
            for rel_id, tokens in enumerate(rel_words):
                for j, word in enumerate(tokens):
                    if j < self.max_rel_words:
                        if word in self.word2id:
                            self.rel_texts[rel_id, j] = self.word2id[word]
                            self.rel_texts_inv[rel_id, j] = self.word2id[word]
                        else:
                            self.rel_texts[rel_id, j] = len(self.word2id)
                            self.rel_texts_inv[rel_id, j] = len(self.word2id)
        else:
            if tokenize == 'bert':
                tokenizer_name = 'bert-base-uncased'
            elif tokenize == 'roberta':
                tokenizer_name = 'roberta-base'
            elif tokenize == 'sbert':
                tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2'
            elif tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif tokenize == 't5':
                tokenizer_name = 't5-small'
            elif tokenize == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words),
                                     pad_val, dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words),
                                         pad_val, dtype=int)

            for rel_id, words in enumerate(rel_words):
                tokens = tokenizer.encode_plus(
                    text=' '.join(words),
                    max_length=self.max_rel_words,
                    pad_to_max_length=True,
                    return_attention_mask=False,
                    truncation=True
                )
                tokens_inv = tokenizer.encode_plus(
                    text=' '.join(words[::-1]),
                    max_length=self.max_rel_words,
                    pad_to_max_length=True,
                    return_attention_mask=False,
                    truncation=True
                )
                self.rel_texts[rel_id] = np.array(tokens['input_ids'])
                self.rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])


class SingleDataLoader(BasicDataLoader):
    def get_batch(self):
        return next(iter(self.data_loader))


def load_dict(file_path):
    with open(file_path, encoding='utf-8') as f:
        return {line.strip(): idx for idx, line in enumerate(f)}


def load_data(config, tokenize):
    print("Load data called...")
    entity2id = load_dict(f"{config['data_folder']}{config['entity2id']}")
    word2id = load_dict(f"{config['data_folder']}{config['word2id']}")
    relation2id = load_dict(f"{config['data_folder']}{config['relation2id']}")

    print("Dictionaries loaded!")

    loaders = {
        data_type: SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type)
        for data_type in ['train', 'dev', 'test']
    }

    return {
        **loaders,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": AutoTokenizer.from_pretrained(tokenize)
    }


if __name__ == "__main__":
    print("data loading! Main function")
    # Define args to avoid undefined name error.
    args = {
        'batch_size': 32,
        'data_folder': './',
        'entity2id': 'entity2id.txt',
        'word2id': 'word2id.txt',
        'relation2id': 'relation2id.txt'
    }
    # Replace `args` with your configuration dictionary as needed.
    dataset = load_data(args, tokenize=lambda x: x)
