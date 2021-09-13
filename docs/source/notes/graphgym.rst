Managing Experiments with GraphGym
==================================

GraphGym is a platform for **designing and evaluating Graph Neural Networks (GNNs)**, as originally proposed in the `"Design Space for Graph Neural Networks" <https://arxiv.org/abs/2011.08843>`__ paper.
We now officially support GraphGym as part of of PyG.

.. warning::

    GraphGym API may change in the future as we are continuously working on better and deeper integration with PyG.

Highlights
----------

#. **Highly modularized pipeline for GNN:**

   - **Data:** Data loading and data splitting
   - **Model:** Modularized GNN implementations
   - **Tasks:** Node-level, edge-level and graph-level tasks
   - **Evaluation:** Accuracy, ROC AUC, ...

#. **Reproducible experiment configuration:**

   - Each experiment is *fully described by a configuration file*

#. **Scalable experiment management:**

   - Easily launch *thousands of GNN experiments in parallel*
   - *Auto-generate* experiment analyses and figures across random seeds and experiments

#. **Flexible user customization:**

   - Easily *register your own modules*, such as data loaders, GNN layers, loss functions, etc

Why GraphGym?
-------------

**TL;DR:** GraphGym is great for GNN beginners, domain experts and GNN researchers.

**Scenario 1:** You are a beginner to graph representation learning and want to understand how GNNs work:

You probably have read many exciting papers on GNNs, and try to write your own GNN implementation.
Even if using raw PyG, you still have to code up the essential pipeline on your own.
GraphGym is a perfect place for your to start learning about *standardized GNN implementation and evaluation*.

.. figure:: ../_figures/graphgym_design_space.png
  :align: center
  :width: 450px

  **Figure 1:** Modularized GNN implementation.

**Scenario 2:** You want to apply GNNs to your exciting application:

You probably know that there are hundreds of possible GNN models, and selecting the best model is notoriously hard.
Even worse, the `GraphGym paper <https://arxiv.org/abs/2011.08843>`__ shows that the best GNN designs for different tasks differ drastically.
GraphGym provides a *simple interface to try out thousands of GNNs in parallel* and understand the best designs for your specific task.
GraphGym also recommends a "go-to" GNN design space, after investigating 10 million GNN model-task combinations.

.. figure:: ../_figures/graphgym_results.png
  :align: center
  :width: 100%

  **Figure 2:** A guideline for desirable GNN design choices.

**Scenario 3:** You are a GNN researcher, who wants to innovate new GNN models or propose new GNN tasks:

Say you have proposed a new GNN layer :class:`ExampleConv`.
GraphGym can help you convincingly argue that :class:`ExampleConv` is better than, *e.g.*, :class:`~torch_geometric.nn.conv.GCNConv`:
When randomly sampling from 10 million possible model-task combinations, how often will :class:`ExampleConv` will outperform :class:`~torch_geometric.nn.conv.GCNConv` when everything else is fixed (including computational costs)?
Moreover, GraphGym can help you easily do hyper-parameter search, and *visualize* what design choices are better.
In sum, GraphGym can greatly facilitate your GNN research.

.. figure:: ../_figures/graphgym_evaluation.png
  :align: center
  :width: 100%

  **Figure 3:** Evaluation of a given GNN design dimension, *e.g.*, :obj:`BatchNorm`.

Basic Usage
-----------

To use GraphGym, you need to clone PyG from Github, then change to the :obj:`graphgym/` directory.

.. code-block:: bash

    git clone https://github.com/pyg-team/pytorch_geometric.git
    cd pytorch_geometric/graphgym

#. **Run a single experiment:**
   Run an experiment using GraphGym via :obj:`run_single.sh`.
   Configurations are specified in :obj:`configs/example_node.yaml`.
   The default experiment is about node classification on the :class:`~torch_geometric.datasets.Planetoid` datasets (using a random 80/20 train/validation split).

   .. code-block:: bash

       bash run_single.sh # run a single experiment

#. **Run a batch of experiments:**
   Run a batch of experiments using GraphGym via :obj:`run_batch.sh`.
   Configurations are specified in :obj:`configs/example_node.yaml` (controls the basic architecture) and :obj:`grids/example.txt` (controls how to do grid search).
   The experiment examines 96 models in the recommended GNN design space, on 2 graph classification datasets.
   Each experiment is repeated 3 times, and we set up that 8 jobs can be concurrently run.
   Depending on your infrastructure, finishing all the experiments may take a long time;
   you can quit the experiment via :obj:`Ctrl-C` (GraphGym will properly kill all the processes).

   .. code-block:: bash

       bash run_batch.sh # run a batch of experiments

#. **Run GraphGym with CPU backend:**
   GraphGym supports CPU backend as well -- you only need to add the line :obj:`device: cpu` to the :obj:`*.yaml` file.
   Here we provide an example:

   .. code-block:: bash

       bash run_single_cpu.sh # run a single experiment using CPU backend

In-Depth Usage
--------------

To use GraphGym, you need to clone PyG from Github, then change to the :obj:`graphgym/` directory.

.. code-block:: bash

    git clone https://github.com/pyg-team/pytorch_geometric.git
    cd pytorch_geometric/graphgym

#. **Run a single experiment:**
   A full example is specified in :obj:`run_single.sh`.

   #. **Specify a configuration file:**
      In GraphGym, an experiment is fully specified by a :obj:`*.yaml` file.
      Unspecified configurations in the :obj:`*.yaml` file will be populated by the default values in :meth:`torch_geometric.graphgym.set_cfg`.
      For example, in :obj:`configs/example_node.yaml`, there are configurations for the dataset, training procedure, model, etc.
      Concrete description for each configuration is described in :meth:`~torch_geometric.graphgym.set_cfg`.

   #. **Launch an experiment:**
      For example, in :obj:`run_single.sh`:

      .. code-block:: bash

          python main.py --cfg configs/example_node.yaml --repeat 3

      You can specify the number of different random seeds to repeat via :obj:`--repeat`.

   #. **Understand the results:**
      Experimental results will be automatically saved in :obj:`results/${CONFIG_NAME}/`.
      In the example above, this amounts to :obj:`results/example_node/`.
      Results for different random seeds will be saved in different subdirectories, *e.g.*, :obj:`results/example_node/2`.
      The aggregated results over all the random seeds are *automatically* generated into :obj:`results/example/agg`, including the mean and standard deviation :obj:`_std` for each metric.
      Train/validation/test results are further saved into subdirectories, such as :obj:`results/example/agg/val`.
      Here, :obj:`stats.json` stores the results after each epoch aggregated across random seeds, and :obj:`best.json` stores the results of *the epoch with the highest validation accuracy*.

#. **Run a batch of experiments:**
   A full example is specified in :obj:`run_batch.sh`.

   #. **Specify a base file:**
      GraphGym supports running a batch of experiments.
      To start, a user needs to select a base architecture via :obj:`--config`.
      The batch of experiments will be created by perturbing certain configurations of the base architecture.

   #. **(Optionally) specify a base file for computational budget:**
      Additionally, GraphGym allows a user to select a base architecture to *control the computational budget* for the grid search via :obj:`--config_budget`.
      The computational budget is currently measured by the number of trainable parameters, and the control is achieved by auto-adjusting the hidden dimensionality of the underlying GNN.
      If no :obj:`--config_budget` is provided, GraphGym will not control the computational budget.

   #. **Specify a grid file:**
      A grid file describes how to perturb the base file, in order to generate the batch of experiments.
      For example, the base file could specify an experiment of a 3-layer GCN for node classification on the :class:`~torch_geometric.datasets.Planetoid` datasets.
      Then, the grid file specifies how to perturb the experiment along different dimensions, such as the number of layers, the model architecture, the dataset, the level of task, etc.

   #. **Generate configuration files for the batch of experiments** based on the information specified above:
      For example, in :obj:`run_batch.sh`:

      .. code-block:: bash

          python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
            --config_budget configs/${DIR}/${CONFIG}.yaml \
            --grid grids/${DIR}/${GRID}.txt \
            --out_dir configs

   #. **Launch the batch of experiments:**
      For example, in :obj:`run_batch.sh`:

      .. code-block:: bash

          bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

      Each experiment will be repeated for :obj:`$REPEAT` times.
      We implemented a queue system to sequentially launch all the jobs, with :obj:`$MAX_JOBS` concurrent jobs running at the same time.
      In practice, our system works great when handling thousands of jobs.

   #. **Understand the results:**
      Experimental results will be automatically saved in directory :obj:`results/${CONFIG_NAME}_grid_${GRID_NAME}/`.
      In the example above, this amounts to :obj:`results/example_grid_example/`.
      After running each experiment, GraphGym additionally automatically averages across different models, saved in :obj:`results/example_grid_example/agg`.
      There, :obj:`val.csv` represents the validation accuracy for each model configuration at the *final* epoch,
      :obj:`val_best.csv` represents the results at the epoch with the highest average validation accuracy, and
      :obj:`val_best_epoch.csv` represents the results at the epoch with the highest validation accuracy averaged over different random seeds.
      When a test set split is provided, :obj:`test.csv` represents the test accuracy for each model configuration at the *final* epoch,
      :obj:`test_best.csv` represents the test set results at the epoch with the highest average validation accuracy, and
      :obj:`test_best_epoch.csv` represents the test set results at the epoch with the highest validation accuracy averaged over different random seeds.

Customizing GraphGym
--------------------

A highlight of GraphGym is that it allows you to easily register customized modules.
For each project, you can have a unique GraphGym copy with different customized modules.
For example, the `"Design Space for Graph Neural Networks" <https://arxiv.org/abs/2011.08843>`__ and `"Identity-aware Graph Neural Networks" <https://arxiv.org/abs/2101.10320>`__ papers represent two successful projects using customized GraphGym, and you may find more details about them `here <https://github.com/snap-stanford/GraphGym#use-case-design-space-for-graph-neural-networks-neurips-2020-spotlight>`__.
Eventually, every GraphGym-powered project will be unique.

There are two ways for customizing GraphGym:

#. Use the :obj:`graphgym/custom_graphgym` directory outside the PyG package:
   You can register your customized modules here without touching PyG. This use case will be great for your own customized project.

#. Use the :obj:`torch_geometric/graphgym/contrib` directory inside the PyG package:
   If you have come up with a nice customized module, you can directly copy your files into :obj:`torch_geometric/graphgym/contrib`, and **create a pull request** to PyG.
   This way, your idea can ship with PyG installations, and will have a much higher visibility and impact.

Concretely, the supported customized modules includes

- Activations: :obj:`custom_graphgym/act/`
- Customized configurations: :obj:`custom_graphgym/config/`
- Feature augmentations: :obj:`custom_graphgym/feature_augment/`
- Feature encoders: :obj:`custom_graphgym/feature_encoder/`
- GNN heads: :obj:`custom_graphgym/head/`
- GNN layers: :obj:`custom_graphgym/layer/`
- Data loaders: :obj:`custom_graphgym/loader/`
- Loss functions: :obj:`custom_graphgym/loss/`
- GNN network architectures: :obj:`custom_graphgym/network/`
- Optimizers: :obj:`custom_graphgym/optimizer/`
- GNN global pooling layers (for graph classification only): :obj:`custom_graphgym/pooling/`
- GNN stages: :obj:`custom_graphgym/stage/`
- GNN training pipelines: :obj:`custom_graphgym/train/`
- Data transformations: :obj:`custom_graphgym/transform/`

Within each directory, at least one example is provided that shows how to register customized modules via :meth:`torch_geometric.graphgym.register`.
Note that new customized modules may result in new configurations.
In these cases, new configuration fields can be registered via :obj:`custom_graphgym/config/`.
