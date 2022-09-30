#!/usr/bin/env bash

# we want one for regular vanilla GCN on ogb-molhiv
python main.py --cfg configs/configs_29.09/vanilla_GCN_2_layer.yaml --repeat 3

# then one for delay GCN on ogb-molhiv
python main.py --cfg configs/configs_29.09/delay_GCN_2_layer.yaml --repeat 3

# vanilla GCN on ogb-molhiv: 5 layers
python main.py --cfg configs/configs_29.09/vanilla_GCN_5_layer.yaml --repeat 3

# delay GCN on ogb-molhiv: 5 layers
python main.py --cfg configs/configs_29.09/delay_GCN_5_layer.yaml --repeat 3

# vanilla GCN on ogb-molhiv: 10 layers
python main.py --cfg configs/configs_29.09/vanilla_GCN_10_layer.yaml --repeat 3

# delay GCN on ogb-molhiv: 10 layers
python main.py --cfg configs/configs_29.09/delay_GCN_10_layer.yaml --repeat 3

# then tensorboards

tensorboard dev upload --logdir ./results/vanilla_GCN_2_layer \
  --name "VanillaGCN 2MP layers" \
  --description "Results for vanilla GCN with GraphGym/OGB defaults (except GCN not GeneralConv and fewer epochs) to compare with DelayGCN version" \
  --one_shot

tensorboard dev upload --logdir ./my_results/delay_GCN_2_layer \
  --name "DelayGCN 2MP layers" \
  --description "Results for delay GCN with GraphGym/OGB defaults" \
  --one_shot

tensorboard dev upload --logdir ./my_results/vanilla_GCN_5_layer \
  --name "VanillaGCN 5MP layers" \
  --description "Results for vanilla GCN with GraphGym/OGB defaults (except GCN not GeneralConv and fewer epochs) to compare with DelayGCN version" \
  --one_shot

tensorboard dev upload --logdir ./my_results/delay_GCN_5_layer \
  --name "DelayGCN 5MP layers" \
  --description "Results for delay GCN with GraphGym/OGB defaults" \
  --one_shot

tensorboard dev upload --logdir ./my_results/vanilla_GCN_10_layer \
  --name "VanillaGCN 10MP layers" \
  --description "Results for vanilla GCN with GraphGym/OGB defaults (except GCN not GeneralConv and fewer epochs) to compare with DelayGCN version" \
  --one_shot

tensorboard dev upload --logdir ./my_results/delay_GCN_10_layer \
  --name "DelayGCN 10MP layers" \
  --description "Results for delay GCN with GraphGym/OGB defaults" \
  --one_shot
