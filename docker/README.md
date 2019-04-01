# Docker

You can run PyTorch Geometric with CUDA 9.0 inside a docker image.
The creation of [our dockerfile](https://github.com/rusty1s/pytorch_geometric/blob/master/docker/Dockerfile) refers to the dockerfiles provided by [NVIDIA](https://gitlab.com/nvidia/cuda/tree/ubuntu16.04) and [PyTorch](https://github.com/anibali/docker-pytorch).

1. Download the dockerfile to your host server.
2. `$ docker build -t "custom image name"`
3. `$ docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "custom image name" /bin/bash`

If you encounter any problems, please feel free to contact <liangshengwen@ict.ac.cn>.
