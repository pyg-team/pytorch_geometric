CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset c10 --model vit-s16 --precision 16 \
                        --label_smoothing 0.1 --mixup 0.8\
                        --train_epochs 600 --sparsity 90\
                        --train_bs 32 --eval_bs 32\
                        --train_lr  1e-3 --train_lr_min 2e-5 --train_decay 5e-2 \
                        --train_graph_lr 5e-2 --train_graph_decay 5e-2\
                        --remark exp1