import argparse

import datasets
from tqdm import tqdm

from torch_geometric.nn.nlp import TXT2KG

# argpars NV_NIM_KEY
parser = argparse.ArgumentParser()
parser.add_argument('--NV_NIM_KEY', type=str, required=True)
args = parser.parse_args()
kg_maker = TXT2KG(
    NVIDIA_API_KEY=args.NV_NIM_KEY,
    chunk_size=512,
)
########
# Use training set for simplicity since our retrieval method is nonparametric
raw_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki')["train"]
# Build KG
num_data_pts = len(raw_dataset)
for idx in tqdm(range(num_data_pts), desc="Building KG"):
    data_point = raw_dataset[idx]
    q = data_point["question"]
    a = data_point["answer"]
    context_doc = ''
    for i in data_point["context"]["sentences"]:
        for sentence in i:
            context_doc += sentence

    kg_maker.add_doc_2_KG(
        txt=context_doc,
        QA_pair=(q, a),
    )
# (TODO) make RAGQueryLoader, need rebase onto Zack's PR

# (TODO) estimate retrieval precision for the training set
"""
approx precision = num_triples_from_a_relevant_doc/num_retrieved_triples
We will use precision as a proxy for recall. This is because for recall,
we must know how many relevant triples exist for each question,
but this is not known.
"""
########
