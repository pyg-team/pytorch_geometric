"""Example workflow for downloading and assembling a multihop QA dataset."""

import argparse
import json
from subprocess import call

import pandas as pd
import torch
import tqdm

from torch_geometric.data import LargeGraphIndexer

# %% [markdown]
# # Encoding A Large Knowledge Graph Part 2

# %% [markdown]
# In this notebook, we will continue where we left off by building a new
# multi-hop QA dataset based on Wikidata.

# %% [markdown]
# ## Example 2: Building a new Dataset from Questions and an already-existing
# Knowledge Graph

# %% [markdown]
# ### Motivation

# %% [markdown]
# One potential application of knowledge graph structural encodings is
# capturing the relationships between different entities that are multiple
# hops apart. This can be challenging for an LLM to recognize from prepended
# graph information. Here's a motivating example (credit to @Rishi Puri):

# %% [markdown]
# In this example, the question can only be answered by reasoning about the
# relationships between the entities in the knowledge graph.

# %% [markdown]
# ### Building a Multi-Hop QA Dataset

# %% [markdown]
# To start, we need to download the raw data of a knowledge graph.
# In this case, we use WikiData5M
# ([Wang et al]
# (https://paperswithcode.com/paper/kepler-a-unified-model-for-knowledge)).
# Here we download the raw triplets and their entity codes. Information about
# this dataset can be found
# [here](https://deepgraphlearning.github.io/project/wikidata5m).

# %% [markdown]
# The following download contains the ID to plaintext mapping for all the
# entities and relations in the knowledge graph:

rv = call("./multihop_download.sh")

# %% [markdown]
# To start, we are going to preprocess the knowledge graph to substitute each
# of the entity/relation codes with their plaintext aliases. This makes it
# easier to use a pre-trained textual encoding model to create triplet
# embeddings, as such a model likely won't understand how to properly embed
# the entity codes.

# %%

# %%
parser = argparse.ArgumentParser(description="Preprocess wikidata5m")
parser.add_argument("--n_triplets", type=int, default=-1)
args = parser.parse_args()

# %%
# Substitute entity codes with their aliases
# Picking the first alias for each entity (rather arbitrarily)
alias_map = {}
rel_alias_map = {}
for line in open('wikidata5m_entity.txt'):
    parts = line.strip().split('\t')
    entity_id = parts[0]
    aliases = parts[1:]
    alias_map[entity_id] = aliases[0]
for line in open('wikidata5m_relation.txt'):
    parts = line.strip().split('\t')
    relation_id = parts[0]
    relation_name = parts[1]
    rel_alias_map[relation_id] = relation_name

# %%
full_graph = []
missing_total = 0
total = 0
limit = None if args.n_triplets == -1 else args.n_triplets
i = 0

for line in tqdm.tqdm(open('wikidata5m_all_triplet.txt')):
    if limit is not None and i >= limit:
        break
    src, rel, dst = line.strip().split('\t')
    if src not in alias_map:
        missing_total += 1
    if dst not in alias_map:
        missing_total += 1
    if rel not in rel_alias_map:
        missing_total += 1
    total += 3
    full_graph.append([
        alias_map.get(src, src),
        rel_alias_map.get(rel, rel),
        alias_map.get(dst, dst)
    ])
    i += 1
print(f"Missing aliases: {missing_total}/{total}")

# %% [markdown]
# Now `full_graph` represents the knowledge graph triplets in
# understandable plaintext.

# %% [markdown]
# Next, we need a set of multi-hop questions that the Knowledge Graph will
# provide us with context for. We utilize a subset of
# [HotPotQA](https://hotpotqa.github.io/)
# ([Yang et. al.](https://arxiv.org/pdf/1809.09600)) called
# [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
# ([Ho et. al.](https://aclanthology.org/2020.coling-main.580.pdf)),
# which includes a subgraph of entities that serve as the ground truth
# justification for answering each multi-hop question:

# %%
with open('train.json') as f:
    train_data = json.load(f)
train_df = pd.DataFrame(train_data)
train_df['split_type'] = 'train'

with open('dev.json') as f:
    dev_data = json.load(f)
dev_df = pd.DataFrame(dev_data)
dev_df['split_type'] = 'dev'

with open('test.json') as f:
    test_data = json.load(f)
test_df = pd.DataFrame(test_data)
test_df['split_type'] = 'test'

df = pd.concat([train_df, dev_df, test_df])

# %% [markdown]
# Now we need to extract the subgraphs

# %%
df['graph_size'] = df['evidences_id'].apply(lambda row: len(row))

# %% [markdown]
# (Optional) We take only questions where the evidence graph is greater than
# 0. (Note: this gets rid of the test set):

# %%
# df = df[df['graph_size'] > 0]

# %%
refined_df = df[[
    '_id', 'question', 'answer', 'split_type', 'evidences_id', 'type',
    'graph_size'
]]

# %% [markdown]
# Checkpoint:

# %%
refined_df.to_csv('wikimultihopqa_refined.csv', index=False)

# %% [markdown]
# Now we need to check that all the entities mentioned in the question/answer
# set are also present in the Wikidata graph:

# %%
relation_map = {}
with open('wikidata5m_relation.txt') as f:
    for line in tqdm.tqdm(f):
        parts = line.strip().split('\t')
        for i in range(1, len(parts)):
            if parts[i] not in relation_map:
                relation_map[parts[i]] = []
            relation_map[parts[i]].append(parts[0])

# %%
entity_set = set()
with open('wikidata5m_entity.txt') as f:
    for line in tqdm.tqdm(f):
        entity_set.add(line.strip().split('\t')[0])

# %%
missing_entities = set()
missing_entity_idx = set()
for i, row in enumerate(refined_df.itertuples()):
    for trip in row.evidences_id:
        entities = trip[0], trip[2]
        for entity in entities:
            if entity not in entity_set:
                # print(
                #    f'The following entity was not found in the KG: {entity}'
                #    )
                missing_entities.add(entity)
                missing_entity_idx.add(i)

# %% [markdown]
# Right now, we drop the missing entity entries. Additional preprocessing can
# be done here to resolve the entity/relation collisions, but that is out of
# the scope for this notebook.

# %%
# missing relations are ok, but missing entities cannot be mapped to
# plaintext, so they should be dropped.
refined_df.reset_index(inplace=True, drop=True)

# %%
cleaned_df = refined_df.drop(missing_entity_idx)

# %% [markdown]
# Now we save the resulting graph and questions/answers dataset:

# %%
cleaned_df.to_csv('wikimultihopqa_cleaned.csv', index=False)

# %%

# %%
torch.save(full_graph, 'wikimultihopqa_full_graph.pt')

# %% [markdown]
# ### Question: How do we extract a contextual subgraph for a given query?

# %% [markdown]
# The chosen retrieval algorithm is a critical component in the pipeline for
# affecting RAG performance. In the next section (1), we will demonstrate a
# naive method of retrieval for a large knowledge graph, and how to apply it
# to this dataset along with WebQSP.

# %% [markdown]
# ### Preparing a Textualized Graph for LLM

# %% [markdown]
# For now however, we need to prepare the graph data to be used as a plaintext
# prefix to the LLM. In order to do this, we want to prompt the LLM to use the
# unique nodes, and unique edge triplets of a given subgraph. In order to do
# this, we prepare a unique indexed node df and edge df for the knowledge
# graph now. This process occurs trivially with the LargeGraphIndexer:

# %%

# %%
indexer = LargeGraphIndexer.from_triplets(full_graph)

# %%
# Node DF
textual_nodes = pd.DataFrame.from_dict(
    {"node_attr": indexer.get_node_features()})
textual_nodes["node_id"] = textual_nodes.index
textual_nodes = textual_nodes[["node_id", "node_attr"]]

# %% [markdown]
# Notice how LargeGraphIndexer ensures that there are no duplicate indices:

# %%
# Edge DF
textual_edges = pd.DataFrame(indexer.get_edge_features(),
                             columns=["src", "edge_attr", "dst"])
textual_edges["src"] = [indexer._nodes[h] for h in textual_edges["src"]]
textual_edges["dst"] = [indexer._nodes[h] for h in textual_edges["dst"]]

# %% [markdown]
# Note: The edge table refers to each node by its index in the node table.
# We will see how this gets utilized later when indexing a subgraph.

# %% [markdown]
# Now we can save the result

# %%
textual_nodes.to_csv('wikimultihopqa_textual_nodes.csv', index=False)
textual_edges.to_csv('wikimultihopqa_textual_edges.csv', index=False)
