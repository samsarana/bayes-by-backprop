# Bayes by Backprop
Reproducibility assessment of the machine learning paper [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf).

We reproduced the results in 2 out of their 3 experiments. We failed to replicate the third experiment, probably because we did not have enough compute to find optimal hyperparameters¹.

### Summary
The key contribution of the paper is the *Bayes by Backprop* algorithm, which the authors claim alleviates two common defects of plain feedforward neural networks: overﬁtting and their inability to assess the uncertainty of their predictions. Furthermore, they claim BBB provides a solution to the exploration vs. exploitation trade-oﬀ in reinforcement learning. The authors run three experiments to provide empirical evidence for these three claims.

We were curious about whether we could reproduce that empirical evidence. Accordingly, this repo contains:
* Code for running the three experiments described in the paper:
  * Experiment 1: `mnist_test.py`
    * `mnist_gridsearch.py` and `mnist_gridsearch_full.py` contain the hyperparameter searches we ran for that experiment.
  * Experiment 2: `regression.py`
  * Experiment 3: `bandits.py`
* A report detailing our findings about whether the paper was reproducible.
* A poster summarising that report.

Additionally, we reran experiment 2 using [*MC-Dropout*](https://arxiv.org/pdf/1506.02142.pdf) instead of Bayes by Backprop, and compared the results. We also did a preliminary investigation into using *Bayes by Backprop* in the active learning framework (see `active_learning_exploration.ipynb`)

---
¹ The authors specified some hyperparameter values, but there were 180 combinations of remaining hyperparameters over which we needed to grid search. A single experiment took 15 hours on our GPU, so we only searched only a small fraction of the possible combinations.
