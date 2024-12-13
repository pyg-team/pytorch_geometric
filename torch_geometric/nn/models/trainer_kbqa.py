import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

tqdm.monitor_iterval = 0

from dataset_load import load_data
from evaluate import Evaluator
from rearev import ReaRev


class Trainer_KBQA:
    """Trainer class for Knowledge-Based Question Answering (KBQA).
    This class handles data loading, model training, evaluation, and checkpoint management.
    """
    def __init__(self, args, model_name, logger=None):
        """Initializes the Trainer_KBQA class with given arguments, model, and logger.

        Args:
            args (dict): Dictionary of training configurations and hyperparameters.
            model_name (str): Name of the model to be used.
            logger (logging.Logger, optional): Logger for logging training and evaluation information. Defaults to None.
        """
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.best_h1b = 0.0
        self.best_f1b = 0.0
        self.eps = args['eps']
        self.warmup_epoch = args['warmup_epoch']
        self.learning_rate = self.args['lr']
        self.test_batch_size = args['test_batch_size']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.reset_time = 0
        self.load_data(args, args['lm'])

        self.decay_rate = args.get('decay_rate', 0.98)

        if model_name == 'ReaRev':
            self.model = ReaRev(self.args, len(self.entity2id),
                                self.num_kb_relation, self.num_word)

        if args['relation_word_emb']:
            self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv)

        self.model.to(self.device)
        self.evaluator = Evaluator(args=args, model=self.model,
                                   entity2id=self.entity2id,
                                   relation2id=self.relation2id,
                                   device=self.device)
        self.load_pretrain()
        self.optim_def()

        self.num_relation = self.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.num_word = len(self.word2id)

        print(
            f"Entity: {self.num_entity}, Relation: {self.num_relation}, Word: {self.num_word}"
        )

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                setattr(self, k,
                        None if v is None else args['data_folder'] + v)

    def optim_def(self):
        """Defines the optimizer and learning rate scheduler for the model.
        """
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim_model = optim.Adam(trainable, lr=self.learning_rate)
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_model, self.decay_rate)

    def load_data(self, args, tokenize):
        """Loads the dataset and initializes related attributes.

        Args:
            args (dict): Training arguments and configurations.
            tokenize (callable): Tokenizer function for processing text.
        """
        dataset = load_data(args, tokenize)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_word = dataset["num_word"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.rel_texts = dataset["rel_texts"]
        self.rel_texts_inv = dataset["rel_texts_inv"]

    def load_pretrain(self):
        """Loads pre-trained weights for the model if specified in arguments.
        """
        args = self.args
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'],
                                     args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=20, write_info=False):
        """Evaluates the model on the given dataset.

        Args:
            data (Dataset): Dataset to evaluate on.
            test_batch_size (int, optional): Batch size for evaluation. Defaults to 20.
            write_info (bool, optional): Whether to write detailed evaluation info. Defaults to False.

        Returns:
            dict: Evaluation metrics including F1, H1, and EM.
        """
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, start_epoch, end_epoch):
        """Trains the model for a given range of epochs.

        Args:
            start_epoch (int): Starting epoch number.
            end_epoch (int): Ending epoch number.
        """
        eval_every = self.args['eval_every']
        print("Start Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()

            if self.decay_rate > 0:
                self.scheduler.step()

            self.logger.info(
                f"Epoch: {epoch + 1}, loss : {loss:.4f}, time: {time.time() - st}"
            )
            self.logger.info(
                f"Training h1 : {np.mean(h1_list_all):.4f}, f1 : {np.mean(f1_list_all):.4f}"
            )

            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1, eval_em = self.evaluate(
                    self.valid_data, self.test_batch_size)
                self.logger.info(
                    f"EVAL F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM {eval_em:.4f}"
                )

                if epoch > self.warmup_epoch:
                    if eval_h1 > self.best_h1:
                        self.best_h1 = eval_h1
                        self.save_ckpt("h1")
                        self.logger.info(f"BEST EVAL H1: {eval_h1:.4f}")
                    if eval_f1 > self.best_f1:
                        self.best_f1 = eval_f1
                        self.save_ckpt("f1")
                        self.logger.info(f"BEST EVAL F1: {eval_f1:.4f}")

                eval_f1, eval_h1, eval_em = self.evaluate(
                    self.test_data, self.test_batch_size)
                self.logger.info(
                    f"TEST F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM {eval_em:.4f}"
                )
        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

    def evaluate_best(self):
        """Evaluates the best saved models (H1, F1, and final).
        """
        filename = os.path.join(
            self.args['checkpoint_dir'],
            "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data,
                                                  self.test_batch_size,
                                                  write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info(
            f"TEST F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM {eval_em:.4f}")

        filename = os.path.join(
            self.args['checkpoint_dir'],
            "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data,
                                                  self.test_batch_size,
                                                  write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info(
            f"TEST F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM {eval_em:.4f}")

        filename = os.path.join(
            self.args['checkpoint_dir'],
            "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data,
                                                  self.test_batch_size,
                                                  write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info(
            f"TEST F1: {eval_f1:.4f}, H1: {eval_h1:.4f}, EM {eval_em:.4f}")

    def evaluate_single(self, filename):
        """Evaluates the model using a single checkpoint.

        Args:
            filename (str): Path to the checkpoint file.
        """
        if filename is not None:
            self.load_ckpt(filename)
        eval_f1, eval_hits, eval_ems = self.evaluate(self.valid_data,
                                                     self.test_batch_size,
                                                     write_info=False)
        self.logger.info(
            f"EVAL F1: {eval_f1:.4f}, H1: {eval_hits:.4f}, EM {eval_ems:.4f}")
        test_f1, test_hits, test_ems = self.evaluate(self.test_data,
                                                     self.test_batch_size,
                                                     write_info=True)
        self.logger.info(
            f"TEST F1: {test_f1:.4f}, H1: {test_hits:.4f}, EM {test_ems:.4f}")

    def train_epoch(self):
        """Trains the model for one epoch.

        Returns:
            tuple: Average loss, extras (placeholders), and lists of H1 and F1 scores.
        """
        self.model.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        h1_list_all = []
        f1_list_all = []
        num_epoch = math.ceil(self.train_data.num_data /
                              self.args['batch_size'])
        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration,
                                              self.args['batch_size'],
                                              self.args['fact_drop'])

            self.optim_model.zero_grad()
            loss, _, _, tp_list = self.model(batch, training=True)
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for name, param in self.model.named_parameters()],
                self.args['gradient_clip'])
            self.optim_model.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        """Saves a checkpoint of the model.

        Args:
            reason (str, optional): Reason for saving the checkpoint (e.g., "h1", "f1"). Defaults to "h1".
        """
        model = self.model
        checkpoint = {'model_state_dict': model.state_dict()}
        model_name = os.path.join(
            self.args['checkpoint_dir'],
            "{}-{}.ckpt".format(self.args['experiment_name'], reason))
        torch.save(checkpoint, model_name)
        print("Best {}, save model as {}".format(reason, model_name))

    def load_ckpt(self, filename):
        """Loads a model checkpoint.

        Args:
            filename (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state_dict, strict=False)
