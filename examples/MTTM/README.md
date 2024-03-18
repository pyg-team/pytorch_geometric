# Multi_type_Transferable_Method
Since the data file uploaded to github is too large, the following two steps need to be run

First, you need to run DataProcessing.py to obtain embedding of node2vec in four datasets
```python
cd Datasets
python DataProcessing.py
```
Second, you need to run MTTM.py
```python
cd ..
python MTTM.py
```



## Citation:

If this code or dataset is useful for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/10004751):

```
@ARTICLE{10004751,
  author={Wang, Huan and Cui, Ziwen and Liu, Ruigang and Fang, Lei and Sha, Ying},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Multi-Type Transferable Method for Missing Link Prediction in Heterogeneous Social Networks}, 
  year={2023},
  volume={35},
  number={11},
  pages={10981-10991},
  keywords={Social networking (online);Feature extraction;Predictive models;Deep learning;Task analysis;Heterogeneous networks;Measurement;Missing link prediction;heterogeneous social network;transferable feature representation},
  doi={10.1109/TKDE.2022.3233481}}
```