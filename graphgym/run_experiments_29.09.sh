#!/usr/bin/env bash

# we want one for regular vanilla GCN on ogb-molhiv
python main.py --cfg configs/configs_29.09/vanilla_GCN_2_layer.yaml --repeat 3

# then tensorboard
tensorboard dev upload --logdir ./results/vanilla_GCN_2_layer \
  --name "VanillaGCN 2MP layers" \
  --description "Results for vanilla GCN with GraphGym/OGB defaults (except GCN not GeneralConv and fewer epochs) to compare with DelayGCN version" \
  --one_shot

# then one for delay GCN on ogb-molhiv
python main.py --cfg configs/configs_29.09/delay_GCN_2_layer.yaml --repeat 3

# then tensorboard
tensorboard dev upload --logdir ./results/delay_GCN_2_layer \
  --name "DelayGCN 2MP layers" \
  --description "Results for delay GCN with GraphGym/OGB defaults" \
  --one_shot

#--- COME TO THESE LATER ---

# then one for regular vanilla GCN with more layers on ogb-molhiv

# then tensorboard

# then one for delay GCN with more layers on ogb-molhiv

