import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
        '--ENDPOINT_URL', type=str, default="https://integrate.api.nvidia.com/v1",
        help="The URL hosting your model, \
        in case you are not using the public NIM.")
    parser.add_argument('--eval_batch_size', type=int,
                        default=2,
                        help="Evaluation batch size")
    parser.add_argument('--llm_generator_name', type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="The LLM to use for Generation")
    parser.add_argument(
        '--llm_generator_mode', type=str, default="full",
        choices=["frozen", "lora",
                 "full"], help="Whether to freeze the Generator LLM,\
                        use LORA, or fully finetune")
    parser.add_argument('--dont_save_model', action="store_true",
                        help="Whether to skip model saving.")
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
    # need a JSON dict of Questions and answers, see below for how its used
    json_path = Path(args.dataset) / "train.json"
    Path(args.dataset) / "corpus"

    with open(json_path) as file:
        json_obj = json.load(file)

    ## TODO: Once generation is completed, this should contain the path to each document for each QA pair
    text_contexts = []

    return json_obj, text_contexts


def make_dataset(args):
    qa_pairs, context_docs = get_data(args)
    print(" ==> Number of Docs:", len(context_docs))
    data_lists = {"test": []}
    global max_chars_in_train_answer
    
    total_data_list = []
    for data_point in tqdm(qa_pairs, desc="Building un-split dataset"):
        if data_point["is_impossible"]:
            continue
        question, answer = data_point["question"], data_point["answer"]
        max_chars_in_train_answer = max(len(answer), max_chars_in_train_answer)

        data = Data()
        data.question = question
        data.label = answer
        # TODO add context
        data.text_context = "dummy contexts for placeholder :)"
        total_data_list.append(data)
    random.shuffle(total_data_list)

    # NOTE: test and validation split have been removed
    data_lists["test"] = total_data_list[int(.8 * len(total_data_list)):]
    dataset_name = os.path.basename(args.dataset)
    dataset_path = os.path.join(args.dataset, f"{dataset_name}.pt")
    torch.save((data_lists, max_chars_in_train_answer), dataset_path)

    return data_lists


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


def test(model, test_loader, args):
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY, args.ENDPOINT_URL)

    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)

    scores = []
    eval_tuples = []
    for iter, test_batch in enumerate(tqdm(test_loader, desc="Testing")):
        if iter > 10:
            break
        new_qs = []
        raw_qs = test_batch["question"]
        for i, q in enumerate(test_batch["question"]):
            # insert VectorRAG context
            new_qs.append(
                prompt_template.format(question=q,
                                       context=test_batch.text_context[i]))

        test_batch.question = new_qs

        ###
        # generator should be given questions with golden contexts
        ###

        preds = (model.inference(question=test_batch.question,
                                 context=test_batch.text_context,
                                 max_tokens=max_chars_in_train_answer))
        for question, pred, label in zip(raw_qs, preds, test_batch.label):
            eval_tuples.append((question, pred, label))
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
    print(f"Starting {args.dataset} training with args: ", args)
    args.NV_NIM_KEY = saved_NIM_KEY

    eval_batch_size = args.eval_batch_size
    data_lists = make_dataset(args)
    test_loader = DataLoader(data_lists["test"], batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    model = get_model(args)
    test(model, test_loader, args)
