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


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)

if __name__ == '__main__':
    main()
