#!/bin/sh

echo "Cora"
echo "===="

echo "GCN"
python gcn.py --dataset=Cora
python gcn.py --dataset=Cora --random_splits

echo "GAT"
python gat.py --dataset=Cora
python gat.py --dataset=Cora --random_splits

echo "Cheby"
python cheb.py --dataset=Cora --num_hops=3
python cheb.py --dataset=Cora --num_hops=3 --random_splits

echo "SGC"
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits

echo "ARMA"
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights
python arma.py --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights --random_splits

echo "APPNP"
python appnp.py --dataset=Cora --alpha=0.1
python appnp.py --dataset=Cora --alpha=0.1 --random_splits

echo "CiteSeer"
echo "========"

echo "GCN"
python gcn.py --dataset=CiteSeer
python gcn.py --dataset=CiteSeer --random_splits

echo "GAT"
python gat.py --dataset=CiteSeer
python gat.py --dataset=CiteSeer --random_splits

echo "Cheby"
python cheb.py --dataset=CiteSeer --num_hops=2
python cheb.py --dataset=CiteSeer --num_hops=3 --random_splits

echo "SGC"
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits

echo "ARMA"
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights --random_splits

echo "APPNP"
python appnp.py --dataset=CiteSeer --alpha=0.1
python appnp.py --dataset=CiteSeer --alpha=0.1 --random_splits

echo "PubMed"
echo "======"

echo "GCN"
python gcn.py --dataset=PubMed
python gcn.py --dataset=PubMed --random_splits

echo "GAT"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits

echo "Cheby"
python cheb.py --dataset=PubMed --num_hops=2
python cheb.py --dataset=PubMed --num_hops=2 --random_splits

echo "SGC"
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits

echo "ARMA"
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits

echo "APPNP"
python appnp.py --dataset=PubMed --alpha=0.1
python appnp.py --dataset=PubMed --alpha=0.1 --random_splits
