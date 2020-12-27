"""Script for running experiments in MDPs.
Using the parameter-based setting by default.
"""
import torch
import numpy as np
import random
import gym
import envs
import argparse
from mellog.logger import Mellogger
import utils
from algorithms.classic_thompson import GaussianDiscreteThompsonSampling
from algorithms.giro import GIRO
from algorithms.phe import PHE
from algorithms.optimist import OPTIMIST
from algorithms.randomistMCMC import RandomistMCMC
from algorithms.randomistMCMC_1step import RandomistMCMC_1Step
from algorithms.ucb1 import UCB1
from algorithms.gpucb import GPUCB


parser = argparse.ArgumentParser(description="Script for running a reinforcement learning experiment.")
# Parameters for any algorithm
parser.add_argument('--n_iterations', type=int, default=5000, help="Number of iterations")
parser.add_argument('--logging_freq', type=int, default=None, help="Logging frequency")
parser.add_argument('--random_seed', type=int, default=torch.randint(low=0, high=(2**32 - 1), size=(1,)).item(),
                    help="Random seed for the experiment")
parser.add_argument('--number_of_policies', type=int, default=49, help="Evaluation frequency")
parser.add_argument('--policy_type', default="linear", choices=["constant", "linear"], help="Policy class to be used")
parser.add_argument('--algorithm', default="randomist",
                    choices={"gaussian", "giro", "phe", "optimist", "ftl", "gpucb", "ucb1",
                             "randomist", "randomistMCMC", "randomistMCMC_1step"},
                    help="Algorithm to be run")
parser.add_argument('--env', default="car", choices={"car", "cartpole"}, help="Environment to be used")
parser.add_argument('--logdir', type=str, required=True, help="Name of the directory to log the results in")
parser.add_argument('--exp_name', type=str, required=True, help="Name of experiment. Should be the same for all runs")
parser.add_argument('--policy_std', type=float, default=0.15, help="Std for Gaussian policies (or hyperpolicies)")
parser.add_argument('--pseudo_rewards_per_timestep', type=float, default=1.1,
                    help="""Number of pseudorewards for unit of history in GIRO, PHE and RANDOMIST.
                         Give negative values for overriding with exploration with random returns.""")
parser.add_argument('--mcmc_steps', type=int, default=10, help="Number of MCMC steps in MCMC-RANDOMIST")
args = vars(parser.parse_args())

# Seed pseudo-random number generators
torch.manual_seed(args['random_seed']), np.random.seed(args['random_seed']), random.seed(args['random_seed'])
# Initialize logger
logger = Mellogger(log_dir=args['logdir'], exp_name=args['exp_name'], args=args, test_mode=False, dump_frequency=10)

# Initialize environment
if args['env'] == "car":
    env = gym.make('MountainCarContinuous-v0')
    parameter_range = torch.tensor([[-1, 1],
                                    [0, 20]]).float()
    return_range = [-5, 95]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    horizon = 999
    args['policy_std'] = torch.tensor([0.15, 3])
elif args['env'] == "cartpole":
    env = gym.make('ContinuousCartpole-v0')
    parameter_range = torch.tensor([[-2, 2],
                                    [0, 4],
                                    [0, 10],
                                    [0, 12]]).float()
    return_range = [0, 200]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    horizon = 200
    args['policy_std'] = torch.tensor([1., 1, 1, 1])
else:
    raise ValueError
# Wrap the environment for pytorch use
env = utils.TorchTensorWrapper(env)
env.seed(args['random_seed'])

# Initialize algorithm
if args['algorithm'] == "gaussian":
    algorithm = GaussianDiscreteThompsonSampling(state_dim=state_dim, action_dim=action_dim,
                                                 number_of_policies=args['number_of_policies'],
                                                 parameter_range=parameter_range,
                                                 policy_std=args['policy_std'], policy_type=args['policy_type'],
                                                 return_range=args['return_range'], setting="parameter_based")
elif args['algorithm'] == "giro":
    algorithm = GIRO(state_dim=state_dim, action_dim=action_dim,
                     number_of_policies=args['number_of_policies'],
                     parameter_range=parameter_range, return_range=return_range,
                     policy_std=args['policy_std'], policy_type=args['policy_type'],
                     pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                     setting="parameter_based")
elif args['algorithm'] == "phe":
    algorithm = PHE(state_dim=state_dim, action_dim=action_dim,
                    number_of_policies=args['number_of_policies'],
                    parameter_range=parameter_range, return_range=return_range,
                    policy_std=args['policy_std'], policy_type="constant",
                    pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                    horizon=horizon, mode='no_is', setting="parameter_based")
elif args['algorithm'] == "optimist":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=args['number_of_policies'],
                         parameter_range=parameter_range, return_range=return_range,
                         policy_std=args['policy_std'], policy_type="constant",
                         horizon=horizon, trajectory_reuse=True,
                         ftl=False, setting="parameter_based")
elif args['algorithm'] == "ftl":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=args['number_of_policies'],
                         parameter_range=parameter_range, return_range=return_range,
                         policy_std=args['policy_std'], policy_type="constant",
                         horizon=horizon, trajectory_reuse=True, ftl=True,
                         setting="parameter_based")
elif args['algorithm'] == "ucb1":
    algorithm = UCB1(state_dim=state_dim, action_dim=action_dim,
                     number_of_policies=args['number_of_policies'],
                     parameter_range=parameter_range, return_range=return_range,
                     policy_std=args['policy_std'], policy_type="constant",
                     horizon=1)
elif args['algorithm'] == "gpucb":
    algorithm = GPUCB(state_dim=state_dim, action_dim=action_dim,
                      number_of_policies=args['number_of_policies'],
                      parameter_range=parameter_range, return_range=return_range,
                      policy_std=args['policy_std'], policy_type="constant",
                      horizon=1)
elif args['algorithm'] == "randomist":
    algorithm = PHE(state_dim=state_dim, action_dim=action_dim,
                    number_of_policies=args['number_of_policies'],
                    parameter_range=parameter_range, return_range=return_range,
                    policy_std=args['policy_std'], policy_type="constant",
                    pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                    horizon=horizon, mode='randomist', setting="parameter_based")
elif args['algorithm'] == "randomistMCMC":
    algorithm = RandomistMCMC(state_dim=state_dim, action_dim=action_dim, parameter_range=parameter_range,
                              policy_std=args['policy_std'], alpha=2, eps=1, a=args['pseudo_rewards_per_timestep'],
                              nMCMCsteps=args['mcmc_steps'], return_range=return_range)
elif args['algorithm'] == "randomistMCMC_1step":
    algorithm = RandomistMCMC_1Step(state_dim=state_dim, action_dim=action_dim, parameter_range=parameter_range,
                                    policy_std=args['policy_std'], alpha=2, eps=1, a=args['pseudo_rewards_per_timestep'],
                                    nMCMCsteps=args['mcmc_steps'], return_range=return_range)
else:
    raise ValueError

total_return = 0
rets = []
for it in range(args['n_iterations']):
    trajectory = utils.collect_trajectory(environment=env, agent=algorithm,
                                          horizon=horizon)
    # Add trajectory to buffer, triggering update of policy
    algorithm.add_to_buffer(trajectory)

    ret = float(torch.sum(torch.tensor(trajectory[2::3])).item())
    total_return += ret
    avg_return = total_return/(it+1)
    logger.log("avg_return", avg_return)
    logger.log("return", ret)
    rets.append(ret)

    if args['logging_freq'] is not None and it % args['logging_freq'] == 0:
        print("[{}/{}] Return={:.2f}, Avg. Return={:.2f}".format(it, args['n_iterations'], ret, avg_return))
