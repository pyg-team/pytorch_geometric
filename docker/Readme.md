Dockerfile
==
# note 

The creation of this dockerfile refers to the code of project [nvidia](https://gitlab.com/nvidia/cuda/tree/ubuntu16.04) and [pytorch](https://github.com/anibali/docker-pytorch).

# How to use?
before using it, please ensure you server with GPU have been install the nvidia-docker
```
1. download this project to your host server.
2. cd pytorch_geometric/docker
3. docker build -t "the name (defined by yourself)" .
4. docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "name defined in step 3" /bin/bash
```

if you have any question, please free contact to me by email: liangshengwen@ict.ac.cn
