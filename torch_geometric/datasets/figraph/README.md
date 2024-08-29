# FiGraph
Our paper **FiGraph: A Dynamic Heterogeneous Graph Dataset for Financial Anomaly Detection**

This paper presents FiGraph, a real-world and large-scale dynamic heterogeneous graph with ground-truth labels for financial anomaly detection. It consists of nine graph snapshots from 2014 to 2022 and comprises 731,914 nodes and 1,044,754 edges. There are five types of nodes and four types of edges.

## **overall.**
Folder 'data' stores the edges data and node attribute data of FiGraph. To fill the above space, this paper presents FiGraph, a real-world and large-scale dynamic heterogeneous graph collected from the Chinese capital market. It consists of 9 graph snapshots from 2014 to 2022 and comprises 730,408 nodes and 1,040,997 edges. Figure 1 illustrates an example of FiGraph. There are mainly five types of nodes: listed companies, unlisted companies, audit institutions, human, and others. The primary goal is to train a GAD model to predict whether a listed company involves a financial fraud. In this specific problem, listed companies are target nodes (TNs), and the rest are background nodes (BNs). Each graph comprises four types of edges: investments, related-party transactions, supply chains, and audit relations. Listed companies exert the most significant impact on the entire capital market and are the focus of regulatory authorities. They are legally required to publish annual financial reports annually in order to be scrutinized for fraudulent activities. Therefore, these nodes have rich attribute information. However, unlisted companies are not obliged to disclose their financial statements, making it challenging to obtain their financial attributes. And the attributes of the remaining nodes are also difficult to obtain in the real world. Therefore, FiGraph is a partially-attributed dynamic heterogeneous graph, where listed company nodes (TNs) possess rich attributes while the remaining nodes lack such information. This dataset can enrich the diversity of current GAD datasets and significantly promote GAD research. 

## Edges
**'edges2014.csv', 'edges2015.csv', 'edges2016.csv', 'edges2017.csv', 'edges2018.csv', 'edges2019.csv', 'edges2020.csv', 'edges2021.csv', 'edges2022.csv'** respectively store one graph snapshot. Each graph snapshot is an undirected graph. Each CSV file consists of three columns, with the first two columns being nodes and the third column being the type of edge. Each row represents an undirected edge.

## Complete Raw Node attributes
**'ListedCompanyFeatures772. csv'** stores the complete raw attributes of all target nodes accross on 9 snapshots. In 'ListedCompanyFeatures772. csv':

* 'nodeID' column is the ID code of the target node, 
* 'Year' represents the year.
* 'Label' represents the label, 1 represents normal while 0 represents fraud.

## Features file and feature selection 
* **ListedCompanyFeatures772.csv'** stores the complete raw attributes of all target nodes and detailed attribute descriptions available in **"FeaturesDescription.xlsx"**.

* **'ListedCompanyFeatures247.csv'** stores the 247 important raw attributes of which 'Cumulative Importance' exceed 80%.

* **feature\_importance.csv** 
'Feature': 575 dimensional attributes. 'Importance': importaance scores calculated by LightGBM.'Normalized_importance': the result obtained by normalizing the importance scores of 575 dimensional features.'Cumulative Importance': All features are arranged in descending order of normalized importance, and the summation score is accumulated starting from top 1.

* **'ListedCompanyFeatures.csv'** is obtained by filling in missing values in 'ListedCompanyFeatures247.csv'.

**Why and how feature selection?**
Target nodes in FiGraph contain extensive and high-dimensional data. We totally extract 772 features for each target node. However, not all features hold equal importance in detecting financial fraud. Additionally, given that financial data often suffers from incomplete reporting and confidentiality constraints, many features exhibit substantial missing values. Therefore, the primary objective of our data cleaning process is to select the most important features and eliminate features with high missing rates.Before feature selection, we remove 197 features with missing values exceeding 30% from the original file, leaving 575. We use median and mode to fill in continuous and discrete variables with missing values, respectively. After taht, we select important features.

Initially, we split the dataset into training and testing sets in an 8:2 ratio using stratified sampling to maintain class distribution. To address class imbalance in the training set, we apply the Synthetic Minority Over-sampling Technique (SMOTE), generating synthetic samples for the minority class to ensure balanced training. We utilize Optuna to optimize hyperparameters of a LightGBM[35] model, aiming to maximize the average precision (AP) score. AP is chosen over AUC due to its sensitivity to class imbalance, providing a more accurate reflection of the model's performance in identifying abnormal instances. We perform 5-fold cross-validation on the training set, which includes an internal split, reserving 20% as a validation set within each fold. This process ensures robust hyperparameter tuning and avoids overfitting. Key hyperparameters tuned include: 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda' and 'scale_pos_weight'. This optimization process spans 50 trials to identify the optimal parameters. After training the LightGBM model with the best parameters on the resampled training set, we calculate feature importance. We retain the top-ranked features that cumulatively account for over 80% of the total importance, balancing model complexity and predictive power. This processing ensures that we select the most important 247 features for the financial fraud detection model. 


# Attribute Description
* Details in **'FeaturesDescription.xlsx'** for all attributes of target nodes.


# License

MIT License

Copyright (c) 2024 Xiaoguang Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
