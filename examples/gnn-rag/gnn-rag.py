import argparse
import os
import time

from torch_geometric.nn.models import Trainer_KBQA
from torch_geometric.utils import create_logger

parser = argparse.ArgumentParser()
add_parse_args(parser)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )

def run_query():
    question = input("Please ask the model a question.")
    query = "Please create a knowledge query for the following question, which leads with one or more relation queries from the question to the answer. " + question

    response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )

    response = response["choices"][0]["message"]["content"].strip()

    answers = ""
    #answers = Iterate knowledge graph

    query = "You want to know: " + question + ". Give a simple answer to the question based on the information provided: " + answers + ". Do not provide any explanation. Only your factual response to the prompt."

    final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )

    final_response = final_response["choices"][0]["message"]["content"].strip()

    return final_response


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name='ReaRev', logger=logger)
    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print(f"Loading pre trained model from {ckpt_path}")
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()

import openai
openai.api_key = "OMITTED"