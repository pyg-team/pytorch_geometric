# PyG in C++

This is a minimal example of getting PyG to work in C++ with CMake.

In order to successfully compile this example, make sure you have both the C++ APIs of [`TorchScatter`](https://github.com/rusty1s/pytorch_scatter#c-api) and [`TorchSparse`](https://github.com/rusty1s/pytorch_sparse/#c-api) installed.

For this, we need to add `TorchLib` to the `-DCMAKE_PREFIX_PATH` (*e.g.*, it may exists in `{CONDA}/lib/python{X.X}/site-packages/torch` if installed via `conda`).
Then, *e.g.*, to install `TorchScatter`, run:

```
https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
mkdir build && cd build
cmake -DWITH_CUDA=on -DCMAKE_PREFIX_PATH="..." ..
make
(sudo) make install
```

Once both dependencies are sorted, we can start the CMake fun:

1. Run `save_model.py` to create and save a PyG GNN model.
2. Create a `build` directory inside the current one.
3. From within the `build` directory, run the following commands:
   * `cmake -DCMAKE_PREFIX_PATH="<PATH_TO_LIBTORCH>;<PATH_TO_TORCHSCATTER>;<PATH_TO_TORCH_SPARSE>" ..`
   * `cmake --build .`

That's it!
You should now have a `hello-world` executable in your `build` folder.
Run it via:

```
./hello-world ../model.pt
```
