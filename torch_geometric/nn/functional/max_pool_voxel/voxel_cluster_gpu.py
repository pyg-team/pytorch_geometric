import torch

from ....utils.cuda import (cuda_num_threads, Stream, Dtype, load_kernel,
                            kernel_loop, get_blocks)

kernel = kernel_loop + """
extern "C"
__global__ void voxel_cluster(
const ${Dtype}* position, const long* cluster_size, long* cluster) {

  CUDA_KERNEL_LOOP(idx, ${num_threads}) {

    ${Dtype} p;
    int c = 0;
    int s;
    int K = ${K};

    for (int d_idx = 0; d_idx < ${dim}; d_idx++) {

        s = cluster_size[d_idx];
        K /= s;

        p = position[idx * ${dim} + d_idx] * s;
        c += min(int(floor(p)), s - 1) * K;
    }

    cluster[idx] = c;
  }
}
"""


def voxel_cluster_gpu(position, cluster_size, K):
    with torch.cuda.device_of(position):
        position = position - position.min(dim=0)[0]
        position /= position.max(dim=0)[0]
        dim = position.size(1)
        cluster = torch.cuda.LongTensor(position.size(0))
        num_threads = cluster.numel()
        f = load_kernel(
            'voxel_cluster',
            kernel,
            Dtype=Dtype(position),
            num_threads=num_threads,
            dim=dim,
            K=K)
        f(args=[
            position.data_ptr(),
            cluster_size.data_ptr(),
            cluster.data_ptr()
        ],
          block=(cuda_num_threads, 1, 1),
          grid=(get_blocks(num_threads), 1, 1),
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return cluster
