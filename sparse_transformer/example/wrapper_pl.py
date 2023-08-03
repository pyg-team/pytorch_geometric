import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
import warmup_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import os

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

class PLWrapper(pl.LightningModule):
    def __init__(self, model, args,cur_iter=0):
        super(PLWrapper, self).__init__()
        self.model = model
        self.num_labels = self.model.num_labels
        self.args = args
        # Debug for the version
        if args.dataset == "c10":
            num_class = 10 
        elif args.dataset == "c100":
            num_class = 100 
        else:
            num_class = 2 
        self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.metric5 = torchmetrics.Accuracy(task='multiclass', num_classes=num_class, top_k = 3)
        self.save_hyperparameters(args)
        self.cur_iter = cur_iter
        if args.mixup > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=self.num_labels)
            self.loss_fn = SoftTargetCrossEntropy()
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing)
        self.eval_criterion = nn.CrossEntropyLoss()

    
    def forward(self, inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        params = list(self.model.named_parameters())
    
        grouped_parameters = [{"params": [p for n, p in params if "param_mask" not in n], 'lr': self.args.train_lr, 'beta':(0.9,0.999), 'weight_decay':self.args.train_decay},
                              {"params": [p for n, p in params if "param_mask" in n], 'lr': self.args.train_graph_lr, 'beta':(0.9,0.999), 'weight_decay':self.args.train_graph_decay}
                             ]
        self.optimizer = torch.optim.AdamW(grouped_parameters, lr=self.args.train_lr, betas=(0.9,0.999), weight_decay=self.args.train_decay)
        self.afterscheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=600, eta_min=self.args.train_lr_min) 
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.args.warmup_epoch, after_scheduler=self.afterscheduler_base)
        return [self.optimizer], [self.scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch+1)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        if self.args.mixup:
            batch['pixel_values'], batch["labels"] = self.mixup_fn(batch['pixel_values'], batch["labels"])
        out = self(batch)
        logits = out["logits"]
        loss = self.loss_fn(logits.view(-1, self.num_labels), batch["labels"])

        acc = self.metric(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.model.vit.encoder.layer[0].attention.attention, 'mask_np'):
            mask_dict = {}
            for i in range(len(self.model.vit.encoder.layer)):
                mask_np = self.model.vit.encoder.layer[i].attention.attention.mask_np
                mask_dict[i] = mask_np
            np.save(os.path.join(self.args.save_folder,"mask_np.npy"), mask_dict)
    
    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, sync_dist=True)

    def _shared_eval(self, batch, batch_idx, prefix):
        labels = batch["labels"]
        out = self(batch)
        logits = out["logits"] if isinstance(out["logits"], torch.Tensor) else  out["logits"][0]
        loss = self.eval_criterion(logits.view(-1, self.num_labels), labels)
        acc = self.metric(logits, labels)
        acc5 = self.metric5(logits, labels)
        self.log("{}_loss".format(prefix), loss, sync_dist=True, prog_bar=True)
        self.log("{}_acc".format(prefix), acc, sync_dist=True, prog_bar=True)
        self.log("{}_acc5".format(prefix), acc5, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")
