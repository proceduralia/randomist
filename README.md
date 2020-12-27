# Policy Optimization as Online Learning with Mediator Feedback
```
@inproceedings{randomist,
  author    = {Alberto Maria Metelli and
               Matteo Papini and
               Pierluca D'Oro and
               Marcello Restelli},
  title     = {Policy Optimization as Online Learning with Mediator Feedback},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021}
}
```

## Installation
We mainly use pytorch and numpy as linear algebra libraries.
Run `pip install -r requirements.txt`, preferably on a clean environment, to install all the required packages.

## Code structure
The structure of the repository is the following:
```
algorithms/
envs/
mellog/
utils.py
...
```
The `algorithms/` directory contains scripts with classes for each algorithm.
Some of them, are grouped together (OPTIMIST and FTL, PHE and the discrete version of RANDOMIST).
`algorithms/randomist.py` and `algorithms/discrete.py` contain the abstract classes instantiated by the other algorithms to perform policy optimization with a discrete number of policies, with and without importance sampling.

## How to run experiments
Experiments can be run with three _main scripts_:
- ``run_my_experiment.py`` for the illustrative experiments in the MAB setting;
- ``run_lqg_experiment.py`` for experiments in the Linear Quadratic Gaussian Regulator environment:
- ``run_rl_experiment.py`` for experiments in the Mountain Car and Cartpole environments. 

Each script requires the specification of a log directory and a name for the experiment. The last one should be the same for different runs of the same experiment.
For each run, a directory is produced, contained files describing the used hyperparameters and a `metrics.csv` file with the logged metrics.
For instance, for running MCMC-RANDOMIST on the Mountain Car environment for 2500 iterations, you can launch:
```
python run_rl_experiment.py --algorithm randomistMCMC --logdir log/mountaincar --exp_name mcmcrandomist --logging_freq 25 --pseudo_rewards_per_timestep 1.1 --env car --n_iterations 2500
```
To reproduce the results in the paper, we also provide three `*.sh` scripts that automate the process of launching multiple runs of the experiments using different algorithms.
For instance, just use the command `sh run_lqg_experiment.sh` to reproduce the complete LQG experiment.
