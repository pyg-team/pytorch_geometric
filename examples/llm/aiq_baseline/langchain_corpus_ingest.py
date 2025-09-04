# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from glob import glob
from uuid import uuid4

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm

from torch_geometric.nn import LLMJudge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants for better readability
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"
# ENCODER_MODEL_NAME_DEFAULT = "Alibaba-NLP/gte-modernbert-base"
ENCODER_MODEL_NAME_DEFAULT = "nvidia/nv-embedqa-e5-v5"
LLM_GENERATOR_NAME_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-ultra-253b-v1"


def emb_text(text_lines, model):
    embeddings = model.encode(text_lines)
    return embeddings


def test(milvus_client, encoder_model, metric_type: str, dataset: str,
         collection_name: str, llm_generator_name: str):

    with open(os.path.join(dataset, "train.json")) as file:
        qa_pairs = json.load(file)

    score = []
    idx = -1
    for data_point in tqdm(qa_pairs, desc="Retrieving pairs"):
        idx += 1
        if idx == 100:
            break
        if data_point.get("is_impossible"):
            continue
        question, answer = (data_point["question"], data_point["answer"])

        print("\nThe question = \n", question, flush=True)
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[
                # Convert the question to an embedding vector
                emb_text(question, encoder_model).tolist()[0]
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=200,  # Return top 200 results
            search_params={
                "metric_type": metric_type,
                "params": {}
            },  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]
        #print(json.dumps(retrieved_lines_with_distances, indent=4))
        #print("\n\n")

        context = "\n".join([
            line_with_distance[0]
            for line_with_distance in retrieved_lines_with_distances
        ])

        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the
        contextual passage snippets provided. Formulate an answer without explanation.
        """

        USER_PROMPT = f"""
        use the following pieces of information enclosed in <context> tags to provide an answer
        to the question enclosed in <question> tags. Enclose the answer within
        <predicted> </predicted> tag. Always try to formulate an answer from the context without
        explanation.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

        # Note: Need to set the env variable 'OPENAI_API_KEY'
        openai_client = OpenAI(base_url=DEFAULT_ENDPOINT_URL)
        response = openai_client.chat.completions.create(
            model=llm_generator_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT
                },
            ],
            #temperature=0.2,
            #top_p=0.7,
            max_tokens=128,
            #stream=True
        )

        # FIXME: Trim response that will be passed to LLM
        #print(response.choices[0].message.content, flush=True)
        response = response.choices[0].message.content
        print(response)
        print("\ncorrect answer: ", answer, flush=True)

        # NOTE: update the env variables for the 'NV_NIM_KEY' and the 'ENDPOINT_URL'

        llm_judge = LLMJudge(NV_NIM_MODEL_DEFAULT, os.getenv('NIM_API_KEY'),
                             DEFAULT_ENDPOINT_URL)
        score.append(llm_judge.score(question, response, answer))
        print("score = ", score, flush=True)
        print("\n\n")

    avg_scores = sum(score) / len(score)
    print("Avg marlin accuracy=", avg_scores, flush=True)
    print("*" * 5 + "NOTE" + "*" * 5, flush=True)
    print("Marlin Accuracy is Estimated by LLM as a Judge!", flush=True)
    print("Improvement of this estimation process is WIP...", flush=True)


async def main(
    *,
    milvus_uri: str,
    collection_name: str,
    dataset: str,
    drop_collection: bool,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")

    # Create the Milvus vector store
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )

    if drop_collection:
        logger.info("Drop Milvus collection: %s", collection_name)
        vector_store.client.drop_collection(collection_name)

    # Check if collection existed (Milvus connects to existing collections during init)
    collection_existed_before = vector_store.col is not None

    if collection_existed_before:  # FIXME
        logger.info("Using existing Milvus collection: %s", collection_name)
        # Get collection info for logging
        try:
            num_entities = vector_store.client.query(
                collection_name=collection_name, filter="",
                output_fields=["count(*)"])
            entity_count = num_entities[0][
                "count(*)"] if num_entities else "unknown number of"
            logger.info("Collection '%s' contains %s documents",
                        collection_name, entity_count)
        except Exception as e:
            logger.warning("Could not get collection info: %s", e)
    else:
        logger.info(
            "Collection '%s' does not exist, will be created when documents are added",
            collection_name)

    dir_to_read = os.path.join(dataset, "corpus")
    files_to_read = os.path.join(dir_to_read, "*.txt")
    text_lines = []
    doc_ids = []
    logger.info("Divide large bodies of text into smaller chunks")
    for file_path in glob(files_to_read, recursive=True):
        with open(file_path) as file:
            file_text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Example chunk size
            chunk_overlap=chunk_overlap,  # Example chunk overlap
            # separators=["\n\n", "\n", " ", ""], # Default separators
            length_function=len,  # Function to measure chunk length
            is_separator_regex=False  # Whether separators are regex patterns
        )

        chunks = text_splitter.split_text(file_text)
        #text_lines += Document(page_content=chunks)
        text_lines.append(Document(page_content=chunks[0]))

    ids = [str(uuid4()) for _ in range(len(text_lines))]
    doc_ids.extend(await vector_store.aadd_documents(documents=text_lines,
                                                     ids=ids))

    return


if __name__ == "__main__":
    import argparse
    import asyncio

    CUDA_COLLECTION_NAME = "qa_docs"
    DEFAULT_URI = "http://localhost:19530"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--milvus_uri", "-u", default=DEFAULT_URI,
                        help="Milvus host URI")
    parser.add_argument("--collection_name", "-n",
                        default=CUDA_COLLECTION_NAME,
                        help="Collection name for the data.")
    parser.add_argument(
        '--dataset', type=str, default="techqa", help="Dataset folder name, "
        "should contain corpus and train.json files."
        "will be saved in the dataset folder")
    parser.add_argument('--llm_generator_name', type=str,
                        default=LLM_GENERATOR_NAME_DEFAULT,
                        help="The LLM to use for Generation")
    parser.add_argument('--drop_collection', action="store_true",
                        help="Drop the collection")
    parser.add_argument('--embedding_model', type=str,
                        default=ENCODER_MODEL_NAME_DEFAULT,
                        help="The embedding model")
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help="Character chunk size when splitting the text")
    parser.add_argument('--chunk_overlap', type=int, default=128,
                        help="Character chunk overlap when splitting the text")
    parser.add_argument('--metric_type', type=str, default="IP",
                        help="Metric type. Other options are COSINE, L2")
    parser.add_argument('--with_react_agent', action="store_true",
                        help="Use react_agent")
    args = parser.parse_args()

    asyncio.run(
        main(milvus_uri=args.milvus_uri, collection_name=args.collection_name,
             dataset=args.dataset, drop_collection=args.drop_collection,
             embedding_model=args.embedding_model, chunk_size=args.chunk_size,
             chunk_overlap=args.chunk_overlap))
