from train_model import Trainer_KBQA
import os 
from parsing import add_parse_args
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    entity_dim: int = 50
    num_epoch: int = 200
    batch_size: int = 8
    eval_every: int = 2
    lm: str = "relbert"
    num_iter: int = 2
    num_ins: int = 3
    num_gnn: int = 3
    name: str = "cwq"
    experiment_name: str = "prn_cwq-rearev-lmsr"
    data_folder: str = "data/CWQ/"
    warmup_epoch: int = 80
    load_experiment = "ReaRev_CWQ.ckpt"

# Example instantiation
config = ExperimentConfig()
print(config)

def main():
    trainer = Trainer_KBQA(args=config, model_name="ReaRev", logger=None)
    trainer.train(0, config.num_epoch - 1)
    ckpt_path = os.path.join(config.data_folder, config.load_experiment)
    print("Loading pre trained model from {}".format(ckpt_path))
    trainer.evaluate_single(ckpt_path)

if __name__ == '__main__':
    main()
