# Bayes by Backprop
Reproducibility assessment of the machine learning paper [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf).

We reproduced the results in 2 out of their 3 experiments. We failed to replicate the third experiment, probably because we did not have enough compute to find optimal hyperparameters¹.

### Summary
The key contribution of the paper is the *Bayes by Backprop* algorithm, which the authors claim alleviates two common defects of plain feedforward neural networks: overﬁtting and their inability to assess the uncertainty of their predictions. Furthermore, they claim *Bayes by Backprop* provides a solution to the exploration vs. exploitation trade-oﬀ in reinforcement learning. The authors run three experiments to provide empirical evidence for these three claims.

We were curious about whether we could reproduce that empirical evidence. Accordingly, this repo contains:
* Code for running the three experiments described in the paper:
  * Experiment 1: [`mnist_test.py`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/mnist_test.py)
    * [`mnist_gridsearch.py`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/mnist_gridsearch.py) and [`mnist_gridsearch_full.py`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/mnist_gridsearch_full.py) contain the hyperparameter searches we ran and would have run (given more compute) for that experiment, respectively.
  * Experiment 2: [`regression.py`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/regression.py)
  * Experiment 3: [`bandits.py`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/bandits.py)
* A [report](https://github.com/samsarana/bayes-by-backprop/blob/master/report.pdf) detailing our findings about whether the paper was reproducible.
* A [poster](https://github.com/samsarana/bayes-by-backprop/blob/master/poster.pdf) summarising that report.

Additionally, we reran experiment 2 using [*MC-Dropout*](https://arxiv.org/pdf/1506.02142.pdf) instead of *Bayes by Backprop*, and compared the results. We also did a preliminary investigation into using *Bayes by Backprop* in the active learning framework (see [`active_learning_exploration.ipynb`](https://github.com/samsarana/bayes-by-backprop/blob/master/code/active_learning_exploration.ipynb))

This work was done by Sam Clarke, Jeffrey Mak and William Lee.

---
¹ The authors specified some hyperparameter values, but there were 180 combinations of remaining hyperparameters over which we needed to grid search. A single experiment took 15 hours on our GPU, so we only searched only a small fraction of the possible combinations.
