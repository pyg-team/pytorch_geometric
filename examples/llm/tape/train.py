import copy
import time
from dataclasses import is_dataclass
from typing import Optional

import pandas as pd
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from tape.config import DatasetName, FeatureType
from tape.dataset.dataset import GraphDataset
from tape.dataset.llm.engine import LlmOfflineEngineArgs, LlmOnlineEngineArgs
from tape.dataset.lm_encoder import LmEncoderArgs
from tape.gnn_model import NodeClassifierArgs
from tape.trainer.gnn_trainer import GnnTrainer, GnnTrainerArgs


def get_parser() -> ArgumentParser:
    # `omegaconf` for variable interpolation
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--dataset', type=DatasetName)
    parser.add_argument('--feature_type', type=FeatureType)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--lm_encoder', type=LmEncoderArgs)
    parser.add_argument('--llm_online_engine',
                        type=Optional[LlmOnlineEngineArgs], default=None)
    parser.add_argument('--llm_offline_engine',
                        type=Optional[LlmOfflineEngineArgs], default=None)
    parser.add_argument('--gnn_model', type=NodeClassifierArgs)
    parser.add_argument('--gnn_trainer', type=GnnTrainerArgs)
    return parser


def update_feature_type(args, feature_type: FeatureType):
    field_name = 'feature_type'
    args_copy = copy.deepcopy(args)
    for attr, attribute_value in vars(args_copy).items():
        if (is_dataclass(attribute_value)
                and hasattr(attribute_value, field_name)):
            field_value = getattr(attribute_value, field_name)
            if (isinstance(field_value, FeatureType)
                    and (field_value == FeatureType.TAPE)):
                setattr(attribute_value, field_name, feature_type)
        elif attr == field_name:
            if (isinstance(attribute_value, FeatureType)
                    and (attribute_value == FeatureType.TAPE)):
                setattr(args_copy, attr, feature_type)
    return args_copy


def _train(args):
    graph_dataset = GraphDataset(
        dataset_name=args.dataset,
        feature_type=args.feature_type,
        lm_encoder_args=args.lm_encoder,
        llm_online_engine_args=args.llm_online_engine,
        llm_offline_engine_args=args.llm_offline_engine,
        device=args.device,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    trainer = GnnTrainer(
        trainer_args=args.gnn_trainer,
        graph_dataset=graph_dataset,
        model_args=args.gnn_model,
    )
    test_output = trainer.train()
    return graph_dataset, test_output


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    start_time = time.perf_counter()

    if args.feature_type == FeatureType.TAPE:
        logits = []
        pred_rows = []
        for value in ('TA', 'P', 'E'):
            ftype = FeatureType._value2member_map_[value]
            _args = update_feature_type(args, feature_type=ftype)
            graph_dataset, test_output = _train(_args)
            logits.append(test_output.logits)
            ftype_str = f'{ftype.name} ({ftype.value})'
            print(f'[Feature type: {ftype_str}] Test accuracy: '
                  f'{test_output.accuracy:.4f}')
            pred_rows.append(
                dict(Feature_type=ftype_str,
                     Test_accuracy=test_output.accuracy))

        # Fuse predictions of features (TA, P, E) by taking an average
        logits = torch.stack(logits).mean(dim=0)
        y_true = graph_dataset.dataset.y
        mask = graph_dataset.dataset.test_mask
        test_acc = GnnTrainer.compute_accuracy(logits=logits, y_true=y_true,
                                               mask=mask)
        ftype_str = f'{args.feature_type.name} ({args.feature_type.value})'
        pred_rows.append(dict(Feature_type=ftype_str,
                              Test_accuracy=test_acc), )

        print()
        print(pd.DataFrame(pred_rows))
    else:
        _, test_output = _train(args)
        ftype_str = f'{args.feature_type.name} ({args.feature_type.value})'
        print(f'[Feature type: {ftype_str}] '
              'Test accuracy: {test_output.accuracy:.4f}')

    execution_time = time.perf_counter() - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f'\nFinished execution in {minutes} minutes and '
          f'{seconds:.2f} seconds.')
