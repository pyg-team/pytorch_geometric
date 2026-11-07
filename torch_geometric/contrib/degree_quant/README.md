## Quickstart
1. Run `pip3 -r requirements.txt` to install necessary dependencies.
2. Run experiment.py to generate results for a few models

## File details
- message_passing.py — Contains `MessagePassingQuant` class. Implementing a new or custom layer requires subclassing from the MPQ The following layers subclass from MPQ:
    - gat_conv.py
    - gcn_conv.py
    - gin_conv.py

- linear.py — Implements `LinearQuantized`, a QAT version of `nn.Linear` layer
- models.py — some predefined models using the different DQ GNN layers
- quantizer.py — implements utilities for quantization (primarily `IntegerQuantizer`)
- utils.py — implements `scatter_` function
- data_handler.py: utility to load datasets and implement transformations
- experiment.py — tests training for a few models
- training.py — defines trianing pipeline
