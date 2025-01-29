from __future__ import annotations

import json
# Import RAGAS
import logging
# redundant
import os
import typing as t
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
# from ragas.exceptions import RagasOutputParserException
from langchain_core.callbacks.base import Callbacks
from langchain_core.prompt_values import StringPromptValue
from langchain_nvidia_ai_endpoints.chat_models import \
    ChatNVIDIA  # Serving the LLM
# from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient, Collection
# from langchain_community.document_loaders import UnstructuredFileLoader
from openai import OpenAI
# Import RAGAS
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class E2E_Accuracy(MetricWithLLM, SingleTurnMetric):
    name: str = field(default="e2e_accuracy", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType,
                              t.Set[str]] = field(default_factory=lambda: {
                                  MetricType.SINGLE_TURN: {
                                      "user_input",
                                      "response",
                                      "reference",
                                  },
                              })
    template_accuracy1 = (
        "Instruction: You are a world class state of the art assistant for rating "
        "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
        "Say 4, if User Answer is full contained and equivalent to Reference Answer"
        "in all terms, topics, numbers, metrics, dates and units.\n"
        "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
        "in all terms, topics, numbers, metrics, dates and units.\n"
        "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
        "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
        "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
        "### Question: {query}\n"
        "### {answer0}: {sentence_inference}\n"
        "### {answer1}: {sentence_true}\n"
        "The rating is:\n")
    template_accuracy2 = (
        "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
        "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
        "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
        "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
        "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
        "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
        "Question: {query}\n\n"
        "{answer0}: {sentence_inference}\n\n"
        "{answer1}: {sentence_true}\n\n"
        "Rating: ")

    def process_score(self, response):
        for i in range(5):
            if str(i) in response[:]:
                return i / 4
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    async def _single_turn_ascore(self, sample: SingleTurnSample,
                                  callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.reference is not None, "Reference is not set"
        assert sample.response is not None, "Response is not set"

        try:
            for retry in range(5):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy1.format(
                        query=sample.user_input,
                        answer0="User Answer",
                        answer1="Reference Answer",
                        sentence_inference=sample.response,
                        sentence_true=sample.reference,
                    ))
                req0 = self.llm.agenerate_text(formatted_prompt, n=1,
                                               temperature=0.10)
                resp0 = await req0
                score_ref_gen = resp0.generations[0][0].text
                score_ref_gen = self.process_score(score_ref_gen)
                if score_ref_gen == score_ref_gen:
                    break
                else:
                    print(f"Retry0: {retry}")

            for retry in range(5):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy2.format(
                        query=sample.user_input,
                        answer0="Reference Answer",
                        answer1="User Answer",
                        sentence_inference=sample.reference,
                        sentence_true=sample.response,
                    ))
                req1 = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.10,
                )
                resp1 = await req1
                score_gen_ref = resp1.generations[0][0].text
                score_gen_ref = self.process_score(score_gen_ref)
                if score_gen_ref == score_gen_ref:
                    break
                else:
                    print(f"Retry1: {retry}")

            score = self.average_scores(score_ref_gen, score_gen_ref)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score

    # For compatibility
    async def _ascore(self, row):
        raise NotImplementedError(
            "You are using deprecated RAGAS version, please update to RAGAS>0.2.6"
        )


os.environ['TOKENIZERS_PARALLELISM'] = "False"
DEVICE = 0  # Specify GPU device
device = torch.device(f"cuda:{DEVICE}")
print(device)

# Enable tqdm for pandas
tqdm.pandas()
if not os.path.exists("results"):
    os.mkdir("results")
# Set your keys if haven't set in env vars.
nvidia_api_key = os.getenv("NVIDIA_API_KEY", 'blank')
hf_access_token = os.getenv("HF_ACCESS_TOKEN", 'blank')
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key,
)

# Retrieval
generator_model_name = "nvdev/meta/llama-3.1-8b-instruct"
remote_embedding_model_name = "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2"
embedding_model_name = remote_embedding_model_name
reranking = False
reranker_model_name = "nvdev/nvidia/llama-3.2-nv-rerankqa-1b-v2"
top_n_for_reranking = 50
top_k = 10


def get_embeddings_with_api(batch, input_type):
    response = client.embeddings.create(
        input=batch, model=embedding_model_name, encoding_format="float",
        extra_body={
            "input_type": input_type,
            "truncate": "END"
        })
    return np.array([d.embedding for d in response.data]).astype(np.float32)


get_embeddings = get_embeddings_with_api

batch_size = 64
if not os.path.exists('./results/tech_qa_embeddings.npy'):
    # Load the chunks df from checkpoint
    print("embedding context chunks")
    import json
    with open('data.json') as file:
        json_obj = json.load(file)
    chunks = []
    for i in json_obj:
        for chunk in i["contexts"]:
            chunks.append(chunk["text"])
    batches = np.array_split(chunks, len(chunks) // batch_size + 1)
    embeddings = []
    for batch in tqdm(batches):
        emb = get_embeddings(batch.tolist(), input_type="passage")
        embeddings.append(emb)

    embeds_text = np.concatenate(embeddings)
    print(f"Chunk embedding shape: {embeds_text.shape}")

    np.save(f'./results/tech_qa_embeddings.npy', embeds_text)
    torch.save(chunks, "./results/tech_qa_context_chunks_as_str.pt")
else:
    # Load chunk embeddings from checkpoint
    embeds_text = np.load(f'./results/tech_qa_embeddings.npy')
    chunks = torch.load("./results/tech_qa_context_chunks_as_str.pt")

df_query = pd.read_json(f"data.json", lines=False)
print(f"Number of queries: {len(df_query)}")

queries = df_query['question'].values
batches = np.array_split(queries, len(queries) // batch_size + 1)

if not os.path.exists('./results/tech_qa_query_embeddings.npy'):
    print("Embedding Queries...")
    embeddings = []
    for batch in tqdm(batches):
        emb = get_embeddings(batch.tolist(), input_type="query")
        embeddings.append(emb)

    embeds_query = np.concatenate(embeddings)
    print(f"Query embedding shape: {embeds_query.shape}")

    np.save(f'./results/tech_qa_query_embeddings.npy', embeds_query)

# Load query embeddings from checkpoint
else:
    embeds_query = np.load(f'./results/tech_qa_query_embeddings.npy')


def rerank(query, passages):
    # dummy
    query = query[:10]
    passages = passages[:3]
    #query = "made up query"
    #passages = ["a", "b", "c", "d", "e"]
    invoke_url = "https://ai.api.nvidia.com/v1/nvdev/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
    }

    payload = {
        "model": reranker_model_name,
        "query": {
            "text": query
        },
        "passages": [{
            "text": p
        } for p in passages],
        "truncate":
        "NONE"  # No truncation, if passage is longer than context window, let it fail explicitly
    }
    # re-use connections
    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    return [x['index'] for x in response_body['rankings']]


def retrieve(query_df, chunks, query_embeddings, chunk_embeddings):
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    # put embeddings on GPU device
    embeds_passage_gpu = torch.from_numpy(chunk_embeddings).to(device)
    embeds_query_gpu = torch.from_numpy(query_embeddings).to(device)

    retrieved_contexts = {}

    if reranking:
        print("Reranking enabled")
    for i in tqdm(range(len(query_df))):
        if query_df.iloc[i]["is_impossible"]:
            continue
        query_embed = embeds_query_gpu[i]

        similarities = cosine(embeds_passage_gpu, query_embed).cpu().numpy()
        # sort similarities from large to small
        ranking = np.argsort(-similarities)
        query_txt = query_df['question'].iloc[i]
        if reranking:
            topN_ids = ranking[:top_n_for_reranking]
            reranked = rerank(query_txt, [chunks[j] for j in topN_ids])
            reranked_ids = topN_ids[reranked]
            ranking = np.concatenate(
                (reranked_ids, ranking[top_n_for_reranking:]))
        topk_in_ranking = ranking[:top_k]
        retrieved_contexts[query_txt] = [chunks[i] for i in topk_in_ranking]
    return retrieved_contexts


embeds_text = np.load(f'./results/tech_qa_embeddings.npy')
embeds_query = np.load(f'./results/tech_qa_query_embeddings.npy')

# Retrievel
if not os.path.exists("./results/retrieved_contexts.pt"):
    retrieved_contexts = retrieve(query_df=df_query, chunks=chunks,
                                  query_embeddings=embeds_query,
                                  chunk_embeddings=embeds_text)
    torch.save(retrieved_contexts, "./results/retrieved_contexts.pt")
else:
    retrieved_contexts = torch.load("./results/retrieved_contexts.pt")
os.environ['TOKENIZERS_PARALLELISM'] = "False"

pd.set_option('display.width', 500)
pd.set_option('max_colwidth', 100)

# Enable tqdm for pandas
tqdm.pandas()


def generate(prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0,
        top_p=1,
        max_tokens=1024,
        stream=False,
    )

    gen_content = completion.choices[0].message.content
    return (gen_content)


# TOO SLOW
# Generating answers:   2%|██▏ | 16/910 [08:07<8:26:46, 34.01s/it]

gen_contents = []
prompt_usage = []
completion_usage = []
# to apples to apples with G-retriever since we cant use such big models yet
# to use such big models need NIM support for G-retriever backend which is future
prompt_template = """Answer this question based on retrieved contexts. Just give the answer without explanation.
[QUESTION]
{question}
[END_QUESTION]

[RETRIEVED_CONTEXTS]
{contexts}
[END_RETRIEVED_CONTEXTS]

Answer: """

if not os.path.exists('./results/final.json'):
    for i, row in tqdm(list(df_query.iterrows()), desc="Generating answers"):
        if row["is_impossible"]:
            gen_content = "-"
        else:
            generated_contexts = retrieved_contexts[row["question"]]
            context_text = "\n".join([chunk for chunk in generated_contexts])
            # print(context_text)
            prompt = prompt_template.format(
                question=row["question"],
                contexts=context_text,
            )
            gen_content = generate(prompt=prompt, model=generator_model_name)
            gen_contents.append(gen_content)

    df_query['generated_answer'] = gen_contents

    df_query.to_json(f'./results/final.json', orient='records', lines=False)
else:
    df_query = pd.read_json(f'./results/final.json', orient='records',
                            lines=False)
print("df_query.tail() =", df_query.tail())
judges = [
    "nvdev/mistralai/mixtral-8x22b-instruct-v0.1",
    "nvdev/meta/llama-3.1-70b-instruct",
    "nvdev/meta/llama-3.3-70b-instruct",
]
trials_per_judge = 2


def verify_columns(df, columns):
    for column in columns:
        if df[column].isna().any() or (df[column].astype(str).str.strip()
                                       == "").any():
            raise ValueError(
                f"The column '{column}' contains NaN or whitespace string values."
            )


from pydantic import ValidationError


# Convert a Pandas DataFrame to RAGAS single turn format
def pandas_to_ragas(df):
    user_input = 'question'
    response_ref = 'answer'
    response_gen = 'generated_answer'

    samples = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row["is_impossible"]:
            continue

        try:
            sample = SingleTurnSample(
                user_input=row[user_input],
                reference=row[response_ref],
                response=row[response_gen],
            )
        except ValidationError:
            print(f"Failed to convert the row: {row}")
            return

        samples.append(sample)

    return EvaluationDataset(samples=samples)


data = df_query
# Check the important fields are not null or empty string.

# Convert to RAGAS Single Turn Sample format
eval_dataset = pandas_to_ragas(
    data.dropna(subset=['question', 'answer', 'generated_answer']))
print(f"Number of samples: {len(eval_dataset)}")

eval_output_dicts = []
for judge in judges:
    print(f"Judging with {judge} ...")
    llm = ChatNVIDIA(
        model=judge,
        nvidia_api_key=nvidia_api_key,
        max_tokens=8,  #  Need to predict only 8 tokens for score.
    )
    evaluator_llm = LangchainLLMWrapper(llm)
    for trial in range(trials_per_judge):
        print(f"  Trial #{trial+1} ...")
        eval_output = evaluate(
            dataset=eval_dataset,
            metrics=[E2E_Accuracy()],
            llm=evaluator_llm,
        )
        print(
            f"  Eval output with judge {judge} trial #{trial+1}: {eval_output}"
        )
        # `eval_output._repr_dict` is a dict with the format of:
        # {'e2e_accuracy': 0.8934426229508197}
        eval_output_dicts.append(eval_output._repr_dict)

eval_outputs_df = pd.DataFrame(eval_output_dicts)
print(f"Eval Avg for {dataset}:")
print(eval_outputs_df.mean())
