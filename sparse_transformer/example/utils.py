import logging
import os

import torch
from datasets import load_dataset
from src.base_sparse_transformer import SequenceModel
from timm.data import create_transform
from transformers.models.vit.configuration_vit import ViTConfig
from wrapper_pl import PLWrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def create_img_transforms(prune_args):
    _train_transforms = create_transform(
        prune_args.img_size, is_training=True,
        color_jitter=prune_args.color_jitter, auto_augment=prune_args.aa,
        re_prob=prune_args.reprob, re_mode=prune_args.remode,
        re_count=prune_args.recount, no_aug=prune_args.no_aug)
    _val_transforms = create_transform(
        prune_args.img_size,
        is_training=False,
    )

    def train_transforms(examples):
        examples['pixel_values'] = [
            _train_transforms(image.convert("RGB"))
            for image in examples['image']
        ]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [
            _val_transforms(image.convert("RGB"))
            for image in examples['image']
        ]
        return examples

    return train_transforms, val_transforms


def create_img_datasets(prune_args):
    if prune_args.dataset == 'c10':
        train_ds, test_ds = load_dataset(
            'cifar10', split=['train[:50000]', 'test[:10000]'])
        train_ds = train_ds.rename_column("img", "image")
        train_ds = train_ds.rename_column("label", "labels")
        test_ds = test_ds.rename_column("img", "image")
        test_ds = test_ds.rename_column("label", "labels")
        val_ds = test_ds

    elif prune_args.dataset == 'c100':
        train_ds, test_ds = load_dataset(
            'cifar100', split=['train[:50000]', 'test[:10000]'])
        train_ds = train_ds.rename_column("img", "image")
        train_ds = train_ds.rename_column("label", "labels")
        test_ds = test_ds.rename_column("img", "image")
        test_ds = test_ds.rename_column("label", "labels")
        val_ds = test_ds

    elif prune_args.dataset == 'imagenet1k':
        train_ds = load_dataset("imagenet-1k", split="train",
                                cache_dir="./ds_imagenet1k")
        test_ds = load_dataset("imagenet-1k", split="validation",
                               cache_dir="./ds_imagenet1k")
        splits = train_ds.train_test_split(test_size=2_0000, shuffle=True,
                                           seed=42, stratify_by_column="label")
        train_ds = splits['train']
        val_ds = splits['test']
    else:
        raise Exception("Dataset not supported!!!")

    train_transforms, val_transforms = create_img_transforms(prune_args)

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    return train_ds, val_ds, test_ds


def create_img_collate_fn(prune_args):
    if prune_args.dataset == "c10":

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    elif prune_args.dataset == "c100":

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples])
            labels = torch.tensor(
                [example["fine_label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    elif prune_args.dataset in ["imagenet1k", "imagenet1ksub"]:

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    return collate_fn


def create_img_kronsp_model(prune_args):
    if prune_args.pretrained:
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        config_name = "config_" + prune_args.model + ".json"
        config_path = os.path.join("./src/configs/", config_name)
        config = ViTConfig.from_pretrained(config_path)
    config.label_smoothing = prune_args.label_smoothing
    config.order = prune_args.order
    config.cond = prune_args.cond
    config.block_list = prune_args.block_list
    config.sparsity = prune_args.sparsity
    config.load_from_pretrained = prune_args.pretrained
    if prune_args.dataset == 'c10':
        config.num_labels = 10
    elif prune_args.dataset == 'c100':
        config.num_labels = 100
    elif prune_args.dataset in ['imagenet1k', 'imagenet1ksub']:
        config.num_labels = 1000
    else:
        raise Exception("Dataset not supported!!!")

    sequence_model = SequenceModel(model="ViT", messages="Kroneckor",
                                   config=config, self_define=False)

    assert prune_args.img_size == config.image_size
    pl_model = PLWrapper(sequence_model.model, prune_args)

    return pl_model, config, sequence_model
