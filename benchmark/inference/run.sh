export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=32
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gcn --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gcn --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gcn --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gat --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gat --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models gat --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models sage --eval-batch-sizes 4096 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models sage --eval-batch-sizes 4096 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models sage --eval-batch-sizes 4096 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models edge_cnn --eval-batch-sizes 4096 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models edge_cnn --eval-batch-sizes 4096 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models edge_cnn --eval-batch-sizes 4096 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models pna --eval-batch-sizes 4096 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models pna --eval-batch-sizes 4096 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets Reddit --models pna --eval-batch-sizes 4096 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gcn --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gcn --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gcn --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gat --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gat --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models gat --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models sage --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models sage --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models sage --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models edge_cnn --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models edge_cnn --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models edge_cnn --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models pna --eval-batch-sizes 8192 --num-layers 1 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models pna --eval-batch-sizes 8192 --num-layers 2 --num-hidden-channels 128 --reuse-device-for-embeddings
numactl --membind 0 --cpunodebind 0 python inference_benchmark.py --device xpu --datasets ogbn-products --models pna --eval-batch-sizes 8192 --num-layers 3 --num-hidden-channels 128 --reuse-device-for-embeddings
