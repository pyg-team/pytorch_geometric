File structure:
- Top level layers in quantized_layers.py (i.e. GINQuant, GCNQuant, LinearQuant etc -- these are exposed in init.py)
- Each layer subclasses from MessagePassingQuant (baseline, ignoring MultiQuant for now...), present in quantized_layers.py (also exposed in __init__.py)
    - Aggregate function of MPQ should use torch_scatter.scatter with reduce parameter = self.aggr
- Each top level layer uses (lq, mpq) created by make_quantizers function available in util.py (exposed)
    - make_quantizer uses create_quantizer, IntegerQuantizer, Quantizer, get_qparams; avaialble in utils.py BUT NOT exposed
- ProbabilisticHighDegree mask defined in utils and exposed, used by Top Level layers


quantized_layers.py:
- Exposed:
    - GINQuant
    - GCNQuant
    - GATQuant
    - LinearQuant
    - MessagePassingQuant (baseline, ignoring Multiquant for now)
        - Aggregate function of MPQ should use torch_scatter.scatter with reduce parameter = self.aggr
        - Aggregate function of MPQ should clamp output of scatter to 10000
            - Why is this hardcoded to 10000?
        - add msg_special_args, aggr_special_args, and update_special_args in MPQ 
        - add REQUIRED_QUANTIZER_KEYS to MPQ, add REQUIRED_X_KEYS inside X class
- Hidden:
    - 

utils.py:
- Exposed:
    - make_quantizer
    - ProbabilisticHighDegree masks
- Hidden:
    - IntegerQuantizer 
        - Contains:
            - Quantizer
            - get_qparams
            - sample_tensor
                - SAMPLE_CUTOFF add as param with default value 1000
                    - Understand what this function does