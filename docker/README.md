# Docker

You can run PyG with CUDA 10.1 inside a docker image.
The creation of [our dockerfile](https://github.com/pyg-team/pytorch_geometric/blob/master/docker/Dockerfile) refers to the dockerfiles provided by [NVIDIA](https://gitlab.com/nvidia/cuda/tree/ubuntu18.04) and [PyTorch](https://github.com/anibali/docker-pytorch).

1. Download the dockerfile to your host server.
1. `$ docker build -t "custom image name"`
1. `$ docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "custom image name" /bin/bash`

If you encounter any problems, please feel free to contact <liangshengwen@ict.ac.cn>.

# Singularity

You can run PyG inside a singularity image. An example singularity file can be found in this folder. You might have to modify the script as versions keep getting updated.

The versions specified in the script were last updated in December 2023. The script installs:

- cuda 12.1 (requires compute capability 5.0 as provided since Maxwell, i.e. Geforce GTX 9xx and newer)
- python 3.11.7 (3.12 was not yet supported by pytorch)
- pytorch 2.1.0 (pytorch 2.1.1 was out but not yet supported by pyg_lib)
- pytorch geometric

## Building and Using the Container

To build the container, run

`sudo singularity build geometric.sif singularity`

then wait.
The container needs to be built on a machine with an NVIDIA GPU in order to install cuda bindings. Otherwise, the CPU version will be installed. (and there is no easy fix to avoid that..)
Once finished, you can run the GAT example in the folder you built the image in by calling

```
wget https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/examples/gat.py
```

(to download the sample),

then

```
singularity exec --nv geometric.sif python3 gat.py
```

to run on the GPU, or

```
singularity exec geometric.sif python3 gat.py
```

in case you have to resort to using the CPU.

## Troubleshooting and additional comments

**Temporary files:** If your harddisk runs full after multiple builds, this is known and apparently working as intended; delete the `/tmp/sbuild-XXXXXXXXX` files.

**Paths:** when running singularity (e.g. `singularity run --nv geometric.sif`) it will load your local .bashrc which can modify you PATH variable. Adding the `--no-home` flag can avoid this behavior.

**Only CPU version:** Check that the `--nv` flag is set when running the container. Also make sure that the machine on which the container is built is equipped with an NVIDIA GPU.

**Deprecation Notice:** NVIDIA only supports the latest CUDA subversion of each docker image. If you receive a deprecation notice, consider increasing the CUDA version of the base docker file from 12.1.1 to 12.1.x.

**Outdated Host NVIDIA driver:** CUDA 12.1 requires a relatively up-to-date nvidia driver on the host. It is not necessary that the exact CUDA versions match as long as the driver is recent enough.

**Compute Capabilities:** If you run it on multiply systems (likely, with singularity), ensure that the device is supported by the CUDA version. If you have the cuda samples installed, check the compute capabilities with (for example) `/usr/local/cuda-10.1/extras/demo_suite/deviceQuery | grep 'CUDA Capability'`; if not, check [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). CUDA 12.x relies com compute capability 5.0 and up, which includes GTX 9xx cards and newer.
