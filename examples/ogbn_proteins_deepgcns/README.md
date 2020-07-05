# ogbn-proteins

We simply apply a random partition to generate batches for both mini-batch training and test. We set the number of partitions to be 10 for training and 5 for test, and we set the batch size to 1 subgraph.  We initialize the features of nodes through aggregating the features of their connected edges by a Sum (Add) aggregation.
## Default 
	--use_gpu False 
    --cluster_number 10 
    --valid_cluster_number 5 
    --aggr add 	#options: [mean, max, add]
    --block plain 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
	--mlp_layers 2
    --norm layer
    --hidden_channels 64
    --epochs 1000
    --lr 0.01
    --dropout 0.0
    --num_evals 1

## DyResGEN-112

### Train the model that performs best
	python main.py --use_gpu --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1 
### Test (use pre-trained model, [download](https://drive.google.com/file/d/1LjsgXZo02WgzpIJe-SQHrbrwEuQl8VQk/view?usp=sharing) from Google Drive)
	python test.py --use_gpu --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1
### Test by multiple evaluations (e.g. 5 times)

    python test.py --use_gpu --num_evals 5 --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1 
    
## Train ResGCN-112
	python main.py --use_gpu --conv_encode_edge --num_layers 112 --block res --gcn_aggr max

#### Train with different GCN models with 28 layers on GPU 

SoftMax aggregator with a learnable t (initialized as 1.0)

    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax --t 1.0 --learn_t

PowerMean aggregator with a learnable p (initialized as 1.0)

    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr power --p 1.0 --learn_p

Apply MsgNorm (message normalization) layer (e.g. SoftMax aggregator with fixed t (e.g. 0.1))

**Not learn parameter s (message scale)**

    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1 --msg_norm
**Learn parameter s (message scale)**

    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1 --msg_norm --learn_msg_scale
    
## ResGEN
SoftMax aggregator with a fixed t (e.g. 0.001)

    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.001
    
PowerMean aggregator with a fixed p (e.g. 5.0)
  
    python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr power --p 5.0
## ResGCN+
	python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr max
## ResGCN 
	python main.py --use_gpu --conv_encode_edge --num_layers 28 --block res --gcn_aggr max
## DenseGCN 
	python main.py --use_gpu --conv_encode_edge --num_layers 7 --block res --gcn_aggr max
## PlainGCN 
	python main.py --use_gpu --conv_encode_edge --num_layers 28 --gcn_aggr max



    
