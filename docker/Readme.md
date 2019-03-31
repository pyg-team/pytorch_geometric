Dockerfile
==
# note 

The creation of this dockerfile refers to the code of project [nvidia](https://gitlab.com/nvidia/cuda/tree/ubuntu16.04) and [pytorch](https://github.com/anibali/docker-pytorch).

# How to use?
```
1. download this project to your host server.
2. cd pytorch_geometric/docker
3. docker build -t "the name (defined by yourself)" .
4. docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "name defined in step 3" /bin/bash
```
