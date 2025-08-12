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

import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from glob import glob
import torch
import torch.nn as nn
from torch_geometric.nn import (
    LLMJudge,
    SentenceTransformer
)

from pymilvus import MilvusClient
from openai import OpenAI
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

ENCODER_MODEL_NAME_DEFAULT = "Alibaba-NLP/gte-modernbert-base"
LLM_GENERATOR_NAME_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"


def main(*,
         milvus_uri: str,
         collection_name: str,
         dataset: str,
         llm_generator_name: str,
         drop_collection: bool,
         embedding_model: str,
         chunk_size: int,
         chunk_overlap: int,
         metric_type: str,
         with_react_agent: bool
         ):

    files_to_read = os.path.join(datasets, "corpus")
    files_to_read = "/home/jnke/tmp/corpus/*.txt"
    text_lines = []
    for file_path in glob(files_to_read, recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        
        logger.info("Divide large bodies of text into smaller chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Example chunk size
            chunk_overlap=chunk_overlap,  # Example chunk overlap
            # separators=["\n\n", "\n", " ", ""], # Default separators
            length_function=len, # Function to measure chunk length
            is_separator_regex=False # Whether separators are regex patterns
        )

        chunks = text_splitter.split_text(file_text)
        text_lines += chunks

    def emb_text(text_lines):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(
            model_name=embedding_model).to(device).eval()
        
        embeddings = model.encode(text_lines)

        return embeddings
    
    test_embedding = emb_text("This is a test")
    # The test above is used to retrieve the embedding dim
    embedding_dim = len(test_embedding[0])

    # Setting the uri as a local file, e.g../milvus.db, is the most convenient method,
    # as it automatically utilizes Milvus Lite to store all data in this file.

    # If you have large scale of data, you can set up a more performant Milvus server on
    # docker or kubernetes. In this setup, please use the server uri, e.g.http://localhost:19530, as your uri.
    # If you want to use Zilliz Cloud, the fully managed cloud service for Milvus, adjust the uri and token,
    # which correspond to the Public Endpoint and Api key in Zilliz Cloud.
    milvus_client = MilvusClient(uri=milvus_uri)


    # FIXME: For now, always drop collection prior to each run
    if drop_collection:
        milvus_client.drop_collection(collection_name)
        logger.info("Successfully dropped the collection '%s'", collection_name)

        # Create a new collection
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            metric_type="IP",  # Inner product distance: 
            # Strong consistency waits for all loads to complete, adding latency with large datasets
            # consistency_level="Strong",  # Strong consistency level
        )
        logger.info("Successfully created collection '%s'", collection_name)

        emb_text_lines = []

        emb_text_lines = emb_text(text_lines).tolist()

        data = [{"id":idx, "vector":text_line, "text":text_lines[idx]} for (idx, text_line) in enumerate(emb_text_lines)]

        """
        data = []
        # logger.info("Adding %s document chunks to Milvus collection %s", len(text_lines), collection_name)
        for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
            data.append({"id": i, "vector": emb_text(line).tolist()[0], "text": line})
        """

        # Insert the data into the collection
        milvus_client.insert(collection_name=collection_name, data=data)
        logger.info("Successfully added %s document chunks to Milvus collection %s", len(text_lines), collection_name)

    if not with_react_agent:    
        # Search the querry in the milvus database
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[
                # Convert the question to an embedding vector
                emb_text(question).tolist()[0]
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=3,  # Return top 3 results
            search_params={"metric_type": metric_type, "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )


        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]
        print(json.dumps(retrieved_lines_with_distances, indent=4))
        print("\n\n")

        context = "\n".join(
            [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
        )

        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """
        USER_PROMPT = f"""
        Only use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
        response = openai_client.chat.completions.create(
            model=llm_generator_name,
            messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
            ],
            #temperature=0.2,
            #top_p=0.7,
            #max_tokens=1024,
            #stream=True
        )

        print(response.choices[0].message.content)
        return response
    

if __name__ == "__main__":
    import argparse

    CUDA_COLLECTION_NAME = "cuda_docs"
    DEFAULT_URI = "http://localhost:19530"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--milvus_uri", "-u", default=DEFAULT_URI, help="Milvus host URI")
    parser.add_argument("--collection_name", "-n", default=CUDA_COLLECTION_NAME, help="Collection name for the data.")
    parser.add_argument(
        '--dataset', type=str, default="techqa", help="Dataset folder name, "
        "should contain corpus and train.json files."
        "will be saved in the dataset folder")
    parser.add_argument(
        '--llm_generator_name', type=str, default=LLM_GENERATOR_NAME_DEFAULT, help="The LLM to use for Generation")
    parser.add_argument(
        '--drop_collection', type=bool, default=True, help="Drop the collection")
    parser.add_argument(
        '--embedding_model', type=str, default=ENCODER_MODEL_NAME_DEFAULT, help="The embedding model")
    parser.add_argument(
        '--chunk_size', type=int, default=1024, help="Character chunk size when splitting the text")
    parser.add_argument(
        '--chunk_overlap', type=int, default=128, help="Character chunk overlap when splitting the text")
    parser.add_argument(
        '--metric_type', type=str, default="IP", help="Metric type. Other options are COSINE, L2")
    parser.add_argument(
        '--with_react_agent', type=bool, default=False, help="Use react_agent")
    args = parser.parse_args()

    main(
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        dataset=args.dataset,
        llm_generator_name=args.llm_generator_name,
        drop_collection=args.drop_collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        metric_type=args.metric_type
    )
