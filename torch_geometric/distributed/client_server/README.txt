Overview
--------

This directory contains an implementation of the PyG feature store behind a
RocksDB instance. RocksDB is an embedded key-value store that leverages sorted
string table (SST) files to store key/value pairs on disk, with aggressive
caching in-memory.

We follow a simple client-server architecture:
  * A server that manages an embedded RocksDB instance is created and
    managed seperately from the model training pipeline. The server
    provides batch-write and batch-read semantics, but does not
    support overlapping reads and writes. Concretely, it is expected
    that `LoadFeatures` is called at initialization to populate
    the RocksDB instance, and subsequently all calls are `Get`
    requests.
  * Every dataloader worker in the model training pipeline initializes
    its own client to communicate with the server. These clients
    are implemented behind the PyG `FeatureStore` interface, and
    issue `Get` requests to fetch relevant features for sampled
    subgraphs to construct `HeteroData` objects for downstream
    processing. These requests are performed over the Google
    RPC (gRPC) framework.

Note that this architecture completely decouples feature storage from training,
such that training can occur on a low-RAM, GPU-enabled instance that
communicates with a high-DRAM, GPU-less instance. There are two axes of
concurrency: the number of DataLoader workers that are created, and the number
of parallel requests that the RocksDB instance can serve. Both are available to
tune.

Build
-----

First, install Apache Arrow (19.0.1) and Google Protobuf (1.66.0):
* https://arrow.apache.org/install/
* https://github.com/grpc/grpc/blob/v1.66.0/src/cpp/README.md

The rest of the implementation build is managed by CMake. To start, from within
the `client_server` directory, initialize CMake with

```
cmake -S . -B build
```

And subsequently build with

```
cd build; make -j .
```

Two artifacts will be produced:
  1. A server that can be started with `./build/server/server`
  2. A python library called `client_server_proto` which is used
     internally by the RocksDBFeatureStore client implementation

You can then create and work with the RocksDBFeatureStore by
importing it within PyG:

```python

from torch_geometric.distributed.client_server import RocksDBFeatureStore
fs = RocksDBFeatureStore(...)
```
