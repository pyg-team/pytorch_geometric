#!/bin/sh

echo "Cora"
echo "===="

echo "GCN"
python gcn.py --dataset=Cora --inference=True
python gcn.py --dataset=Cora --random_splits=True --inference=True
python gcn.py --dataset=Cora --inference=True --profile=True
python gcn.py --dataset=Cora --random_splits=True --inference=True --profile=True

echo "GAT"
python gat.py --dataset=Cora --inference=True
python gat.py --dataset=Cora --random_splits=True --inference=True
python gat.py --dataset=Cora --inference=True --profile=True
python gat.py --dataset=Cora --random_splits=True --inference=True --profile=True

echo "Cheby"
python cheb.py --dataset=Cora --num_hops=3 --inference=True
python cheb.py --dataset=Cora --num_hops=3 --random_splits=True --inference=True
python cheb.py --dataset=Cora --num_hops=3 --inference=True --profile=True
python cheb.py --dataset=Cora --num_hops=3 --random_splits=True --inference=True --profile=True

echo "SGC"
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --inference=True
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits=True --inference=True
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --inference=True --profile=True
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits=True --inference=True --profile=True

echo "ARMA"
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --inference=True
python arma.py --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True --inference=True
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --inference=True --profile=True
python arma.py --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True --inference=True --profile=True

echo "APPNP"
python appnp.py --dataset=Cora --alpha=0.1 --inference=True
python appnp.py --dataset=Cora --alpha=0.1 --random_splits=True --inference=True
python appnp.py --dataset=Cora --alpha=0.1 --inference=True --profile=True
python appnp.py --dataset=Cora --alpha=0.1 --random_splits=True --inference=True --profile=True

echo "CiteSeer"
echo "========"

echo "GCN"
python gcn.py --dataset=CiteSeer --inference=True
python gcn.py --dataset=CiteSeer --random_splits=True --inference=True
python gcn.py --dataset=CiteSeer --inference=True --profile=True
python gcn.py --dataset=CiteSeer --random_splits=True --inference=True --profile=True

echo "GAT"
python gat.py --dataset=CiteSeer --inference=True
python gat.py --dataset=CiteSeer --random_splits=True --inference=True
python gat.py --dataset=CiteSeer --inference=True --profile=True
python gat.py --dataset=CiteSeer --random_splits=True --inference=True --profile=True

echo "Cheby"
python cheb.py --dataset=CiteSeer --num_hops=2 --inference=True
python cheb.py --dataset=CiteSeer --num_hops=3 --random_splits=True --inference=True
python cheb.py --dataset=CiteSeer --num_hops=2 --inference=True --profile=True
python cheb.py --dataset=CiteSeer --num_hops=3 --random_splits=True --inference=True --profile=True

echo "SGC"
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --inference=True
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits=True --inference=True
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --inference=True --profile=True
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits=True --inference=True --profile=True

echo "ARMA"
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --inference=True
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True --inference=True
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --inference=True --profile=True
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True --inference=True --profile=True

echo "APPNP"
python appnp.py --dataset=CiteSeer --alpha=0.1 --inference=True
python appnp.py --dataset=CiteSeer --alpha=0.1 --random_splits=True --inference=True
python appnp.py --dataset=CiteSeer --alpha=0.1 --inference=True --profile=True
python appnp.py --dataset=CiteSeer --alpha=0.1 --random_splits=True --inference=True --profile=True

echo "PubMed"
echo "======"

echo "GCN"
python gcn.py --dataset=PubMed --inference=True
python gcn.py --dataset=PubMed --random_splits=True --inference=True
python gcn.py --dataset=PubMed --inference=True --profile=True
python gcn.py --dataset=PubMed --random_splits=True --inference=True --profile=True

echo "GAT"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --inference=True
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True --inference=True
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --inference=True --profile=True
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True --inference=True --profile=True

echo "Cheby"
python cheb.py --dataset=PubMed --num_hops=2 --inference=True
python cheb.py --dataset=PubMed --num_hops=2 --random_splits=True --inference=True
python cheb.py --dataset=PubMed --num_hops=2 --inference=True --profile=True
python cheb.py --dataset=PubMed --num_hops=2 --random_splits=True --inference=True --profile=True

echo "SGC"
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --inference=True
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits=True --inference=True
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --inference=True --profile=True
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits=True --inference=True --profile=True

echo "ARMA"
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --inference=True
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits=True --inference=True
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --inference=True --profile=True
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits=True --inference=True --profile=True

echo "APPNP"
python appnp.py --dataset=PubMed --alpha=0.1 --inference=True
python appnp.py --dataset=PubMed --alpha=0.1 --random_splits=True --inference=True
python appnp.py --dataset=PubMed --alpha=0.1 --inference=True --profile=True
python appnp.py --dataset=PubMed --alpha=0.1 --random_splits=True --inference=True --profile=True
