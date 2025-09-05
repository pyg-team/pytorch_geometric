import argparse
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import LLM, LLMJudge

max_chars_in_train_answer = 128
sys_prompt = (
    "You are an expert assistant that can answer "
    "any question from its knowledge, given a knowledge graph embedding and "
    "it's textualized context. Just give the answer, without explanation.")
prompt_template = """
    [QUESTION]
    {question}
    [END_QUESTION]

    [RETRIEVED_CONTEXTS]
    {context}
    [END_RETRIEVED_CONTEXTS]
    """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                        help="The NIM LLM to use for LLMJudge")
    parser.add_argument('--NV_NIM_KEY', type=str, help="NVIDIA API key")
    parser.add_argument(
        '--ENDPOINT_URL', type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="The URL hosting your model, \
        in case you are not using the public NIM.")
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument('--llm_generator_name', type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="The LLM to use for Generation")
    parser.add_argument(
        '--llm_generator_mode', type=str, default="full",
        choices=["frozen", "lora",
                 "full"], help="Whether to freeze the Generator LLM,\
                        use LORA, or fully finetune")
    parser.add_argument(
        '--num_gpus', type=int, default=None,
        help="Number of GPUs to use. If not specified,"
        "will determine automatically based on model size.")
    parser.add_argument(
        '--dataset', type=str, default="techqa", help="Dataset folder name, "
        "should contain corpus and train.json files. extracted triples, "
        "processed dataset, document retriever, and model checkpoints will "
        "be saved in the dataset folder")
    args = parser.parse_args()

    assert args.NV_NIM_KEY, "NVIDIA API key is required for TXT2KG and eval"

    return args


def get_data(args):
    json_path = Path(args.dataset) / "train.json"
    with open(json_path) as file:
        return json.load(file)


def make_dataset(args):
    print("Reading in QA Data...")
    qa_data = get_data(args)
    print(" ==> Number of Docs:", len(qa_data))

    global max_chars_in_train_answer
    total_data_list = []
    for pair in tqdm(qa_data, desc="Building un-split dataset"):
        max_chars_in_train_answer = max(len(pair['answer']),
                                        max_chars_in_train_answer)
        data = Data()
        data.question = pair['question']
        data.label = pair['answer']
        data.context_doc = pair['file_name']
        total_data_list.append(data)
    random.shuffle(total_data_list)

    dataset_name = os.path.basename(args.dataset)
    dataset_path = os.path.join(args.dataset, f"{dataset_name}.pt")
    torch.save((total_data_list, max_chars_in_train_answer), dataset_path)

    return total_data_list


def get_model(args):
    if args.llm_generator_mode == "full":
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  n_gpus=args.num_gpus)
    elif args.llm_generator_mode == "lora":
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  dtype=torch.float32, n_gpus=args.num_gpus)
    else:
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  dtype=torch.float32, n_gpus=args.num_gpus).eval()
        for _, p in llm.named_parameters():
            p.requires_grad = False

    return llm


def test(model, data_list, args):
    print(f"LLMJudge using {args.NV_NIM_MODEL}")
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY, args.ENDPOINT_URL)

    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)

    scores = []
    eval_tuples = []
    for iter, test_batch in enumerate(tqdm(data_list, desc="Testing")):
        # if iter > 3:
        #     print("Ending early for debugging")
        #     break
        q_with_context = ""
        context = ""
        # TODO: should this be done in a different way?
        # insert VectorRAG context
        doc_path = Path(args.dataset) / "corpus" / test_batch.context_doc
        with open(doc_path) as f:
            context = f.read()
            q_with_context = prompt_template.format(
                question=test_batch.question, context=context)

        # LLM generator inference step
        # TODO: please check if this makes sense. context was "" in txt2kg_rag. Setting it equals to the context doc returned some weird output. A list[~260]
        qs = [
            q_with_context,
        ]

        pred = model.inference(question=qs,
                               max_tokens=max_chars_in_train_answer / 3)

        eval_tuples.append((q, pred, test_batch.label))
    for question, pred, label in tqdm(eval_tuples, desc="Eval"):
        scores.append(eval(question, pred, label))

    avg_scores = sum(scores) / len(scores)
    print("Avg marlin accuracy=", avg_scores)
    print("*" * 5 + "NOTE" + "*" * 5)
    print("Marlin Accuracy is Estimated by LLM as a Judge!")
    print("Improvement of this estimation process is WIP...")


if __name__ == '__main__':
    # for reproducibility
    seed_everything(50)
    args = parse_args()

    # Need to sanitize sensitive keys
    saved_NIM_KEY = args.NV_NIM_KEY
    args.NV_NIM_KEY = "********"
    print(
        f"Starting {args.dataset} training with args:\n{json.dumps(args, indent=4, sort_keys=True)}"
    )
    args.NV_NIM_KEY = saved_NIM_KEY

    eval_batch_size = args.eval_batch_size
    data_lists = make_dataset(args)

    model = get_model(args)
    test(model, data_lists[:200], args)
