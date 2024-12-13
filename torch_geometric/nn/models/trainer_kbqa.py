import os
import time

import numpy as np
import torch
from dataset_load import load_data
from evaluate import Evaluator
from rearev import ReaRev
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class TrainerKBQA:
    """Trainer for Knowledge-Based Question Answering
    (KBQA) using PyTorch and PyG.

    Handles data loading, model training, evaluation,
    and checkpoint management.
    """
    def __init__(self, args, model_name, logger=None):
        """Initialize Trainer with configuration, model, and logger.

        Args:
            args (dict): Training configurations and hyperparameters.
            model_name (str): Name of the model to use.
            logger (logging.Logger, optional): Logger for training logs.
        """
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if args["use_cuda"] else "cpu")

        # Hyperparameters and training settings
        self.learning_rate = args["lr"]
        self.decay_rate = args.get("decay_rate", 0.98)
        self.test_batch_size = args["test_batch_size"]
        self.warmup_epoch = args["warmup_epoch"]

        # Data loading
        self.dataset = load_data(args, args["lm"])
        self.train_data = self.dataset["train"]
        self.valid_data = self.dataset["valid"]
        self.test_data = self.dataset["test"]
        self.entity2id = self.dataset["entity2id"]
        self.relation2id = self.dataset["relation2id"]
        self.word2id = self.dataset["word2id"]
        self.rel_texts = self.dataset.get("rel_texts")
        self.rel_texts_inv = self.dataset.get("rel_texts_inv")
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_word = len(self.word2id)

        # Model initialization
        if model_name == "ReaRev":
            self.model = ReaRev(
                args,
                num_entity=self.num_entity,
                num_relation=self.num_relation,
                num_word=self.num_word,
            )
        if args.get("relation_word_emb"):
            self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv)

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.decay_rate)
        self.evaluator = Evaluator(
            args=args,
            model=self.model,
            entity2id=self.entity2id,
            relation2id=self.relation2id,
            device=self.device,
        )

        # Load pretrained weights if specified
        if args.get("load_experiment"):
            self.load_checkpoint(args["load_experiment"])

    def train(self, start_epoch, end_epoch):
        """Train the model over a range of epochs.

        Args:
            start_epoch (int): Starting epoch.
            end_epoch (int): Ending epoch.
        """
        eval_every = self.args["eval_every"]
        for epoch in range(start_epoch, end_epoch + 1):
            start_time = time.time()

            loss, h1_list, f1_list = self._train_epoch()
            if self.decay_rate > 0:
                self.scheduler.step()

            avg_h1, avg_f1 = np.mean(h1_list), np.mean(f1_list)
            self.logger.info(
                f"Epoch {epoch + 1}, Loss: {loss:.4f},
                Time: {time.time() - start_time:.2f}s"
            )
            self.logger.info(f"Training H1: {avg_h1:.4f}, F1: {avg_f1:.4f}")

            # Evaluation
            if (epoch + 1) % eval_every == 0:
                eval_metrics = self.evaluate(self.valid_data)
                eval_f1, eval_h1, eval_em = eval_metrics["f1"], eval_metrics[
                    "h1"], eval_metrics["em"]
                self.logger.info(
                    f"Validation - F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM: {eval_em:.4f}"
                )

                if epoch > self.warmup_epoch:
                    if eval_h1 > getattr(self, "best_h1", 0):
                        self.best_h1 = eval_h1
                        self.save_checkpoint(f"best-h1-epoch-{epoch}")
                    if eval_f1 > getattr(self, "best_f1", 0):
                        self.best_f1 = eval_f1
                        self.save_checkpoint(f"best-f1-epoch-{epoch}")

    def _train_epoch(self):
        """Train the model for one epoch.

        Returns:
            tuple: Average loss, H1 scores, and F1 scores.
        """
        self.model.train()
        losses, h1_list, f1_list = [], [], []
        data_loader = self.train_data.data_loader(
            batch_size=self.args["batch_size"], shuffle=True)

        for batch in tqdm(data_loader, desc="Training"):
            self.optimizer.zero_grad()
            loss, _, _, tp_list = self.model(batch, training=True)
            h1_scores, f1_scores = tp_list

            loss.backward()
            clip_grad_norm_(self.model.parameters(),
                            self.args["gradient_clip"])
            self.optimizer.step()

            losses.append(loss.item())
            h1_list.extend(h1_scores)
            f1_list.extend(f1_scores)

        return np.mean(losses), h1_list, f1_list

    def evaluate(self, data):
        """Evaluate the model on a dataset.

        Args:
            data (Dataset): Dataset to evaluate.

        Returns:
            dict: Evaluation metrics (F1, H1, EM).
        """
        self.model.eval()
        with torch.no_grad():
            return self.evaluator.evaluate(data,
                                           batch_size=self.test_batch_size)

    def save_checkpoint(self, name):
        """Save model checkpoint.

        Args:
            name (str): Name of the checkpoint.
        """
        checkpoint_path = os.path.join(self.args["checkpoint_dir"],
                                       f"{name}.ckpt")
        torch.save({"model_state_dict": self.model.state_dict()},
                   checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, name):
        """Load model checkpoint.

        Args:
            name (str): Name of the checkpoint to load.
        """
        checkpoint_path = os.path.join(self.args["checkpoint_dir"],
                                       f"{name}.ckpt")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"],
                                   strict=False)
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
