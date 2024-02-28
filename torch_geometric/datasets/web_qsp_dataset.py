import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from torch_geometric.data import InMemoryDataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(torch.nn.Module):
    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids,
                                   attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def sbert_text2embedding(model, tokenizer, device, text):
    try:
        encoding = tokenizer(text, padding=True, truncation=True,
                             return_tensors='pt')
        dataset = Dataset(input_ids=encoding.input_ids,
                          attention_mask=encoding.attention_mask)

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        # Placeholder for storing the embeddings
        all_embeddings = []

        # Iterate through batches
        with torch.no_grad():

            for batch in dataloader:
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(input_ids=batch["input_ids"],
                                   att_mask=batch["att_mask"])

                # Append the embeddings to the list
                all_embeddings.append(embeddings)

        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    except:  # noqa
        print("SBERT text embedding failed, returning 0s...")
        return torch.zeros((0, 1024))

    return all_embeddings


class WebQSPDataset(InMemoryDataset):
    r"""The WebQuestionsSP dataset was released as part of
    “The Value of Semantic Parse Labeling for Knowledge
    Base Question Answering”
    [Yih, Richardson, Meek, Chang & Suh, 2016].
    It contains semantic parses, vs. answers, for a set of questions
    that originally comes from WebQuestions [Berant et al., 2013]."
    Processing based on "G-Retriever: Retrieval-Augmented Generation
    for Textual Graph Understanding and Question Answering".
    Requires datasets and transformers from HuggingFace.

    Args:
        root (str): Root directory where the dataset should be saved.
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        root: str,
        force_reload: bool = False,
    ) -> None:

        super().__init__(root, None, None, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.model_name = 'sbert'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def download(self) -> None:
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.raw_dataset = datasets.concatenate_datasets(
            [dataset['train'], dataset['validation'], dataset['test']])
        self.split_idxs = {
            'train':
            torch.arange(len(dataset['train'])),
            'val':
            torch.arange(len(dataset['validation'])) + len(dataset['train']),
            'test':
            torch.arange(len(dataset['test'])) + len(dataset['train']) +
            len(dataset['validation'])
        }

    def process(self) -> None:
        self.questions = [i['question'] for i in self.raw_dataset]
        pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
        self.model = Sentence_Transformer(pretrained_repo)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)
        self.text2embedding = sbert_text2embedding
        # encode questions
        print('Encoding questions...')
        self.q_embs = self.text2embedding(self.model, self.tokenizer,
                                          self.device, self.questions)
        print('Encoding graphs...')
        list_of_graphs = []
        for index in tqdm(range(len(self.raw_dataset))):
            raw_nodes = {}
            raw_edges = []
            for tri in self.raw_dataset[i]['graph']:
                h, r, t = tri
                h = h.lower()
                t = t.lower()
                if h not in raw_nodes:
                    raw_nodes[h] = len(raw_nodes)
                if t not in raw_nodes:
                    raw_nodes[t] = len(raw_nodes)
                raw_edges.append({
                    'src': raw_nodes[h],
                    'edge_attr': r,
                    'dst': raw_nodes[t]
                })
            nodes = pd.DataFrame([{
                'node_id': v,
                'node_attr': k
            } for k, v in nodes.items()], columns=['node_id', 'node_attr'])
            edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])
            # encode nodes
            nodes.node_attr.fillna("", inplace=True)
            x = self.text2embedding(self.model, self.tokenizer, self.device,
                                    nodes.node_attr.tolist())
            # encode edges
            edge_attr = self.text2embedding(self.model, self.tokenizer,
                                            self.device,
                                            edges.edge_attr.tolist())
            edge_index = torch.LongTensor(
                [edges.src.tolist(), edges.dst.tolist()])
            list_of_graphs.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     num_nodes=len(nodes), q_emb=self.q_embs[i]))
        self.save(list_of_graphs, self.processed_paths[0])
