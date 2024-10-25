import argparse

import datasets
from tqdm import tqdm

from torch_geometric.nn.nlp import TXT2KG

parser = argparse.ArgumentParser()
parser.add_argument('--NV_NIM_KEY', type=str, required=True)
parser.add_argument('--percent_data', type=int, default=10)
args = parser.parse_args()
assert args.percent_data <= 100 and args.percent_data > 0
kg_maker = TXT2KG(
    NVIDIA_API_KEY=args.NV_NIM_KEY,
    chunk_size=512,
)

# Use training set for simplicity since our retrieval method is nonparametric
raw_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki')["train"]
# Build KG
num_data_pts = len(raw_dataset)
data_idxs = torch.randperm(num_data_pts)[0:int(num_data_pts *
                                               float(args.percent_data) /
                                               100.0)]
for idx in tqdm(data_idxs, desc="Building KG"):
    data_point = raw_dataset[idx]
    q = data_point["question"]
    a = data_point["answer"]
    context_doc = ''
    for i in data_point["context"]["sentences"]:
        for sentence in i:
            context_doc += sentence

    QA_pair = (q, a)
    kg_maker.add_doc_2_KG(
        txt=context_doc,
        QA_pair=QA_pair,
    )
# (TODO) make RAGQueryLoader from kg_maker, need rebase onto Zack's PR
# based on example:
# https://github.com/pyg-team/pytorch_geometric/blob/f607374fc8250e5f08b10e82e8ada2adf2ed18cc/examples/llm/g_retriever_utils/rag_generate.py
"""
approx precision = num_relevant_out_of_retrieved/num_retrieved_triples
We will use precision as a proxy for recall. This is because for recall,
we must know how many relevant triples exist for each question,
but this is not known.
"""
precisions = []
for QA_pair in kg_maker.relevant_triples.keys()
    relevant_triples = kg_maker.relevant_triples[QA_pair]
    retrieved_triples = #(TODO) call RAGQueryLoader
    num_relevant_out_of_retrieved = float(sum([retrieved_triple in relevant_triples for retrieved_triple in retrieved_triples]))
    precisions.append(num_relevant_out_of_retrieved/len(retrieved_triples))
approx_precision = mean(precisions)
