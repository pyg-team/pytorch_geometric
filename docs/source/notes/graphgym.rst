GraphGym - Manager for experiments
==================================

**Author: Jiaxuan You**

GraphGym is a platform for **designing and evaluating Graph Neural Networks (GNNs)**.
GraphGym is originally proposed in `Design Space for Graph Neural Networks, NeurIPS 2020 Spotlight <https://arxiv.org/abs/2011.08843>`__ and hosted in `Github repository <https://github.com/snap-stanford/GraphGym>`__.
GraphGym has teamed up with PyTorch Geometric to provide an easy-to-use workflow for GNN development.
GraphGym API may change in the future as we are continuously working on better and deeper integration with PyG;
in the meantime, the current version has many nice functionalities and has shown to be successful in many applications.

Highlights
----------

**1. Highly modularized pipeline for GNN**

- **Data:** Data loading, data splitting
- **Model:** Modularized GNN implementation
- **Tasks:** Node / edge / graph level GNN tasks
- **Evaluation:** Accuracy, ROC AUC, ...

**2. Reproducible experiment configuration**

- Each experiment is *fully described by a configuration file*

**3. Scalable experiment management**

- Easily launch *thousands of GNN experiments in parallel*
- *Auto-generate* experiment analyses and figures across random seeds and experiments.

**4. Flexible user customization**

- Easily *register your own modules*, such as data loaders, GNN layers, loss functions, etc.

Why GraphGym?
-------------

**TL;DR:** GraphGym is great for GNN beginners, domain experts and GNN researchers.

**Scenario 1:** You are a beginner to GNNs, who wants to understand how GNNs work.

You probably have read many exciting papers on GNNs, and try to write your own GNN implementation.
Even if using raw PyTorch Geometric, you still have to code up the essential pipeline on your own.
GraphGym is a perfect place for your to start learning *standardized GNN implementation and evaluation*.


.. figure:: ../_figures/graphgym_design_space.png
  :align: center
  :width: 400px

  Figure 1: Modularized GNN implementation


**Scenario 2:** You want to apply GNN to your exciting applications.


You probably know that there are hundreds of possible GNN models, and selecting the best model is notoriously hard.
Even worse, we have shown in our `paper <https://arxiv.org/abs/2011.08843>`__ that the best GNN designs for different tasks differ drastically.
GraphGym provides a *simple interface to try out thousands of GNNs in parallel* and understand the best designs for your specific task.
GraphGym also recommends a "go-to" GNN design space, after investigating 10 million GNN model-task combinations.


.. figure:: ../_figures/graphgym_results.png
  :align: center
  :width: 700px

  Figure 2: A guideline for desirable GNN design choices
  (Sampling from 10 million GNN model-task combinations)


**Scenario 3:** You are a GNN researcher, who wants to innovate GNN models / propose new GNN tasks.

Say you have proposed a new GNN layer :obj:`ExampleConv`.
GraphGym can help you convincingly argue that :obj:`ExampleConv` is better than say :obj:`GCNConv`:
when randomly sample from 10 million possible model-task combinations, how often :obj:`ExampleConv` will outperform :obj:`GCNConv`,
when everything else is fixed (including the computational cost).
Moreover, GraphGym can help you easily do hyper-parameter search, and *visualize* what design choices are better.
In sum, GraphGym can greatly facilitate your GNN research.

.. figure:: ../_figures/graphgym_evaluation.png
  :align: center
  :width: 700px

  Figure 3: Evaluation of a given GNN design dimension
  (BatchNorm here)


Basic Usage
-----------

To use GraphGym, you need to first clone the lastest PyG from Github, then change to :obj:`graphgym` directory.

.. code-block:: bash

    git clone https://github.com/rusty1s/pytorch_geometric.git
    cd pytorch_geometric/graphgym


**1 Run a single experiment.**
Run a test GNN experiment using GraphGym :obj:`run_single.sh`.
Configurations are specified in :obj:`configs/example_node.yaml`.
The experiment is about node classification on Cora dataset (random 80/20 train/val split).

.. code-block:: bash

    bash run_single.sh # run a single experiment

**2 Run a batch of experiments.**
Run a batch of GNN experiments using GraphGym :obj:`run_batch.sh`.
Configurations are specified in :obj:`configs/example_node.yaml` (controls the basic architecture)
and :obj:`grids/example.txt` (controls how to do grid search).
The experiment examines 96 models in the recommended GNN design space, on 2 graph classification datasets.
Each experiment is repeated 3 times, and we set that 8 jobs can be concurrently run.
Depending on your infrastructure, finishing all the experiments may take a long time;
you can quit the experiment by :obj:`Ctrl-C` (GraphGym will properly kill all the processes).

.. code-block:: bash

    bash run_batch.sh # run a batch of experiments

**3 Run GraphGym with CPU backend.**
GraphGym supports cpu backend as well -- you only need to add one line :obj:`device: cpu` to the :obj:`.yaml` file. Here we provide an example.

.. code-block:: bash

    bash run_single_cpu.sh # run a single experiment using CPU backend




GraphGym In-depth Usage
-----------------------

To use GraphGym, you need to first clone the lastest PyG from Github, then change to :obj:`graphgym` directory.

.. code-block:: bash

    git clone https://github.com/rusty1s/pytorch_geometric.git
    cd pytorch_geometric/graphgym

**1 Run a single GNN experiment**
A full example is specified in :obj:`run_single.sh`.

**1.1 Specify a configuration file.**
In GraphGym, an experiment is fully specified by a :obj:`.yaml` file.
Unspecified configurations in the :obj:`.yaml` file will be populated by the default values in
:meth:`~torch_geometric.graphgym.set_cfg`.
For example, in :obj:`configs/example_node.yaml`,
there are configurations on dataset, training, model, GNN, etc.
Concrete description for each configuration is described in
:meth:`~torch_geometric.graphgym.set_cfg`.

**1.2 Launch an experiment.**
For example, in :obj:`run_single.sh`:

.. code-block:: bash

    python main.py --cfg configs/example_node.yaml --repeat 3

You can specify the number of different random seeds to repeat via :obj:`--repeat`.

**1.3 Understand the results.**
Experimental results will be automatically saved in directory :obj:`results/${CONFIG_NAME}/`;
in the example above, it is :obj:`results/example_node/`.
Results for different random seeds will be saved in different subdirectories, such as :obj:`results/example/2`.
The aggregated results over all the random seeds are *automatically* generated into :obj:`results/example/agg`,
including the mean and standard deviation :obj:`_std` for each metric.
Train/val/test results are further saved into subdirectories, such as :obj:`results/example/agg/val`; here,
:obj:`stats.json` stores the results after each epoch aggregated across random seeds,
:obj:`best.json` stores the results at *the epoch with the highest validation accuracy*.

**2 Run a batch of GNN experiments**
A full example is specified in :obj:`run_batch.sh`.

**2.1 Specify a base file.**
GraphGym supports running a batch of experiments.
To start, a user needs to select a base architecture :obj:`--config`.
The batch of experiments will be created by perturbing certain configurations of the base architecture.

**2.2 (Optional) Specify a base file for computational budget.**
Additionally, GraphGym allows a user to select a base architecture to *control the computational budget* for the grid search, :obj:`--config_budget`.
The computational budget is currently measured by the number of trainable parameters; the control is achieved by auto-adjust
the hidden dimension size for GNN.
If no :obj:`--config_budget` is provided, GraphGym will not control the computational budget.

**2.3 Specify a grid file.**
A grid file describes how to perturb the base file, in order to generate the batch of the experiments.
For example, the base file could specify an experiment of 3-layer GCN for Cora node classification.
Then, the grid file specifies how to perturb the experiment along different dimension, such as number of layers,
model architecture, dataset, level of task, etc.


**2.4 Generate config files for the batch of experiments,** based on the information specified above.
For example, in :obj:`run_batch.sh`:

.. code-block:: bash

    python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
      --config_budget configs/${DIR}/${CONFIG}.yaml \
      --grid grids/${DIR}/${GRID}.txt \
      --out_dir configs

**2.5 Launch the batch of experiments.**
For example, in :obj:`run_batch.sh`:
.. code-block:: bash

    bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

Each experiment will be repeated for :obj:`$REPEAT` times.
We implemented a queue system to sequentially launch all the jobs, with :obj:`$MAX_JOBS` concurrent jobs running at the same time.
In practice, our system works great when handling thousands of jobs.

**2.6 Understand the results.**
Experimental results will be automatically saved in directory :obj:`results/${CONFIG_NAME}_grid_${GRID_NAME}/`;
in the example above, it is :obj:`results/example_grid_example/`.
After running each experiment, GraphGym additionally automatically averages across different models, saved in
:obj:`results/example_grid_example/agg`.
There, :obj:`val.csv` represents validation accuracy for each model configuration at the *final* epoch;
:obj:`val_best.csv` represents the results at the epoch with the highest average validation error;
:obj:`val_best_epoch.csv` represents the results at the epoch with the highest validation error, averaged over different random seeds.
When test set split is provided, :obj:`test.csv` represents test accuracy for each model configuration at the *final* epoch;
:obj:`test_best.csv` represents the test set results at the epoch with the highest average validation error;
:obj:`test_best_epoch.csv` represents the test set results at the epoch with the highest validation error, averaged over different random seeds.






Customize your GraphGym
-----------------------
A highlight of GraphGym is that it allows you to easily register your customized modules.
For each project, you could have a unique GraphGym copy with different customized modules.
For example, `Design Space for Graph Neural Networks <https://arxiv.org/abs/2011.08843>`__ and `Identity-aware Graph Neural Networks <https://arxiv.org/abs/2101.10320>`__
are two successful projects using customized GraphGym, and you may find more details `here <https://github.com/snap-stanford/GraphGym#use-case-design-space-for-graph-neural-networks-neurips-2020-spotlight>`__.
Eventually, every GraphGym-powered project will be unique :)

There are two ways for customizing GraphGym:

- Use :obj:`graphgym/custom_graphgym` directory which is outside the PyG package. You may register your customized modules here, without touching PyG package. This use case will be great for your own customized project.
- Use :obj:`torch_geometric/graphgym/contrib`. If you have come up with a nice customized module, you can directly copy your file into :obj:`torch_geometric/graphgym/contrib`, and **create a pull request** to PyG. This way, your idea can ship with PyG installations, and will have a much higher visibility and impact.

Concretely, the supported customized modules includes

- Activation :obj:`custom_graphgym/act/`
- Customized configurations :obj:`custom_graphgym/config/`
- Feature augmentation :obj:`custom_graphgym/feature_augment/`
- Feature encoder :obj:`custom_graphgym/feature_encoder/`
- GNN head :obj:`custom_graphgym/head/`
- GNN layer :obj:`custom_graphgym/layer/`
- Data loader :obj:`custom_graphgym/loader/`
- Loss function :obj:`custom_graphgym/loss/`
- GNN network architecture :obj:`custom_graphgym/network/`
- Optimizer :obj:`custom_graphgym/optimizer/`
- GNN global pooling (graph classification only) :obj:`custom_graphgym/pooling/`
- GNN stage :obj:`custom_graphgym/stage/`
- GNN training pipeline :obj:`custom_graphgym/train/`
- Data transformations :obj:`custom_graphgym/transform/`

Within each directory, at least an example is provided, showing how to register user customized modules via :meth:`torch_geometric.graphgym.register`.
Note that new user customized modules may result in new configurations; in these cases, new configuration fields
can be registered at :obj:`custom_graphgym/config/`.

As we have mentioned earlier, we welcome you to move your customized module into :obj:`torch_geometric/graphgym/contrib`, and create a pull request for us, so that your ideas can contribute to the whole community.

**Note: Applying to your own datasets.**
A common use case will be applying GraphGym to your favorite datasets. To do so, you may follow our example in
:obj:`custom_graphgym/loader/example.py` to register your favorite loaders.


