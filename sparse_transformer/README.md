# Sparse Graph Transformer

This repository implements a versatile API framework for sparse transformers using efficient message-passing mechanisms. The framework's cornerstone is the computation of `sparse_messages`, which significantly accelerates attention computation. This framework can seamlessly integrate with the popular PyTorch Lightning package, and we provide a comprehensive example within this codebase.


## Codebase Overview

The `base_sparse_transformer` module implements a generic SequenceModel that can accept any form of sequence model. As an optimal choice, we recommend using a backbone model from the HuggingFace library. To illustrate this, we've implemented the Kroneckor-based message passing scheme and a ViT-based backbone model. All the requirements for running the source code are documented in `requirements.txt`.

To run the current example, navigate to the example directory and execute the provided shell script:

```shell
bash run.sh
```
