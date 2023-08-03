
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import create_img_datasets, create_img_collate_fn
from utils import create_img_kronsp_model, create_img_kronatt_model
parser = argparse.ArgumentParser()

parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--dataset", default="c10", type=str, help="c10 | c100 | imgnet")
parser.add_argument("--no_aug", action="store_true")
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                        "(default: rand-m9-mstd0.5-inc1)')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',  help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',help='Color jitter factor (default: 0.4)')
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--img_size", default=224, type=int, help="img_size")
parser.add_argument("--remark", type=str, default="",  help="remarks")
parser.add_argument("--save_folder", type=str, default="",  help="save folder")
parser.add_argument("--label_smoothing", default=0, type=float, help="smoothing")
parser.add_argument("--mixup", default=0, type=float, help="mixup")
parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument("--train_bs", default=512, type=int, help="training epochs count")
parser.add_argument("--eval_bs", default=512, type=int, help="training epochs count")
# parser.add_argument("--train_bs", default=2, type=int, help="training epochs count")
# parser.add_argument("--eval_bs", default=2, type=int, help="training epochs count")

parser.add_argument("--train_epochs", default=300, type=int, help="training epochs count")
parser.add_argument("--warmup_epoch", default=5, type=int, help="warm up")
parser.add_argument("--train_lr", default=3e-3, type=float, help="learning rate")
parser.add_argument("--train_lr_min", default=1e-5, type=float, help="learning rate")
parser.add_argument("--train_decay", default=1e-4, type=float, help="weight decay")
parser.add_argument("--train_graph_lr", default=3e-3, type=float, help="learning rate")#2e-5 for fine-tune
parser.add_argument("--train_graph_decay", default=1e-4, type=float, help="weight decay")

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--model_impl", default="dgl", type=str, help="torch|dgl|xformers")
parser.add_argument("--model", default="vit-s16", type=str)
parser.add_argument("--model_config", default="config_vit-s16", type=str)

parser.add_argument("--sparsity", default=90, type=float, help="Sparsity level")
parser.add_argument("--order", default=2, type=int, help="Kronecker decomposition order")
parser.add_argument("--block_list", nargs='+', default=[14, 14], help='decomposition block size', type=int)
parser.add_argument("--decompose_att", action="store_true")
parser.add_argument("--cond", action="store_true")
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()
args.devices = torch.cuda.device_count()
args.num_workers =  2*args.devices
if len(args.save_folder) == 0:
    model_name = "kronatt" if args.decompose_att else "kronsp"
    folder_name = "{}_".format(args.dataset)+ model_name +"{}_order{}".format(args.order, args.model)
    if len(args.remark)>0:
        folder_name += "_"
        folder_name += args.remark
    args.save_folder = os.path.join("./save/"+model_name, folder_name)

assert args.order == len(args.block_list)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

pl.utilities.seed.seed_everything(42)

train_ds, val_ds, test_ds = create_img_datasets(args)
collate_fn = create_img_collate_fn(args)

# typical_data = train_ds[0]
# for typical_data in train_ds:
#     print(typical_data['labels'])
# exit()

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True,collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_bs, num_workers=args.num_workers, pin_memory=True,collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_bs, num_workers=args.num_workers, pin_memory=True,collate_fn=collate_fn)

if args.decompose_att:
    pl_model, config = create_img_kronatt_model(args)
else:
    pl_model, config = create_img_kronsp_model(args)

output_dir = args.save_folder
checkpoint_callback = ModelCheckpoint(dirpath=output_dir, 
                                monitor="val_acc",
                                mode = "max",
                                save_weights_only=False,
                                save_last = True,
                                save_top_k=1, 
                                auto_insert_metric_name = True,
                                every_n_epochs = 1, #default
)
cb_list = [checkpoint_callback]

logger = pl.loggers.CSVLogger(save_dir=output_dir)

trainer = pl.Trainer(precision=args.precision,
            accelerator="gpu",
            devices = args.devices,
            strategy="ddp",
            check_val_every_n_epoch=1,
            logger=logger, 
            max_epochs=args.train_epochs, 
            callbacks=cb_list
)

if args.resume:
    outputs = trainer.fit(pl_model, ckpt_path=os.path.join(args.save_folder,"last.ckpt"),train_dataloaders = train_dl, val_dataloaders = test_dl)
else:
    outputs = trainer.fit(pl_model, train_dataloaders = train_dl, val_dataloaders = test_dl)

