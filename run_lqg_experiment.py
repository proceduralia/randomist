import torch
import numpy as np
import random
import argparse
from mellog.logger import Mellogger
from envs.mab import LQGMAB
from algorithms.classic_thompson import GaussianDiscreteThompsonSampling
from algorithms.giro import GIRO
from algorithms.phe import PHE
from algorithms.optimist import OPTIMIST
from algorithms.ucb1 import UCB1
from algorithms.gpucb import GPUCB


parser = argparse.ArgumentParser(description="Script for running a mab experiment.")
# Parameters for any algorithm
parser.add_argument('--n_iterations', type=int, default=5000, help="Number of iterations")
parser.add_argument('--logging_freq', type=int, default=None, help="Logging frequency")
parser.add_argument('--evaluate_every', type=int, default=1, help="Evaluation frequency")
parser.add_argument('--random_seed', type=int, default=torch.randint(low=0, high=(2**32 - 1), size=(1,)).item(),
                    help="Random seed for the experiment")
parser.add_argument('--number_of_policies', type=int, default=49, help="Evaluation frequency")
parser.add_argument('--algorithm', default="randomist",
                    choices={"gaussian", "giro", "phe", "optimist", "ftl",
                             "gpucb", "ucb1", "randomist"},
                    help="Algorithm to be run")
parser.add_argument('--env', default="ackley", choices={"ackley", "ackley5", "rastrigin", "leon"}, help="Environment to be used")
parser.add_argument('--logdir', type=str, required=True, help="Name of the directory to log3 the results in")
parser.add_argument('--exp_name', type=str, required=True, help="Name of experiment. Should be the same for all runs")
parser.add_argument('--policy_std', type=float, default=1.5, help="Std for Gaussian policies")
parser.add_argument('--pseudo_rewards_per_timestep', type=float, default=0.1,
                    help="""Number of pseudorewards for unit of history in GIRO, PHE and RANDOMIST.""")
args = vars(parser.parse_args())

# Seed pseudo-random number generators
torch.manual_seed(args['random_seed']), np.random.seed(args['random_seed']), random.seed(args['random_seed'])
# Initialize logger
logger = Mellogger(log_dir=args['logdir'], exp_name=args['exp_name'], args=args, test_mode=False)

# Initialize environment

mab = LQGMAB()
parameter_range = [-1, 1]
return_range = [0, 1]
state_dim = 1
action_dim = 1

# Initialize algorithm
if args['algorithm'] == "gaussian":
    algorithm = GaussianDiscreteThompsonSampling(state_dim=state_dim, action_dim=action_dim,
                                                 number_of_policies=args['number_of_policies'],
                                                 parameter_range=parameter_range,
                                                 policy_std=args['policy_std'], policy_type="constant",
                                                 return_range=return_range, setting="parameter_based")
elif args['algorithm'] == "giro":
    algorithm = GIRO(state_dim=state_dim, action_dim=action_dim,
                     number_of_policies=args['number_of_policies'],
                     parameter_range=parameter_range, return_range=return_range,
                     policy_std=args['policy_std'], policy_type="constant",
                     pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                     setting="parameter_based")
elif args['algorithm'] == "phe":
    algorithm = PHE(state_dim=state_dim, action_dim=action_dim,
                    number_of_policies=args['number_of_policies'],
                    parameter_range=parameter_range, return_range=return_range,
                    policy_std=args['policy_std'], policy_type="constant",
                    pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                    horizon=1, mode='no_is', setting="parameter_based")
elif args['algorithm'] == "optimist":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=args['number_of_policies'],
                         parameter_range=parameter_range, return_range=return_range,
                         policy_std=args['policy_std'], policy_type="constant",
                         horizon=1, trajectory_reuse=True,
                         ftl=False, setting="parameter_based")
elif args['algorithm'] == "ftl":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=args['number_of_policies'],
                         parameter_range=parameter_range, return_range=return_range,
                         policy_std=args['policy_std'], policy_type="constant",
                         horizon=1, trajectory_reuse=True, ftl=True,
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
                    horizon=1, mode='randomist', setting="parameter_based")
else:
    raise ValueError

dummy_state = torch.randn(1)
training_rewards = []
cumulative_regret = 0
optimal_arm_value = 1
for it in range(args['n_iterations']):
    # Collect a new interaction
    action = algorithm(dummy_state, greedy=False)
    reward = mab(action)
    training_rewards.append(reward)

    # Add interaction to buffer, triggering update of policy
    trajectory = [dummy_state, action, reward]
    algorithm.add_to_buffer(trajectory)

    if it % args['evaluate_every'] == 0:
        # Compute regret with the not-greedy policy
        action = algorithm(dummy_state, greedy=False, evaluation_mode=True)
        mab.make_deterministic()
        reward = mab(action)
        mab.make_stochastic()
        instantaneous_regret = optimal_arm_value - reward
        cumulative_regret = cumulative_regret + instantaneous_regret
        logger.log("policy_index", algorithm.current_policy_index)
        logger.log("evaluation_actions", action.numpy())
        logger.log("evaluation_rewards", reward.item())
        logger.log("cumulative_regret", cumulative_regret.item())
    if args['logging_freq'] is not None and it % args['logging_freq'] == 0:
        print("[{}/{}] Regret={}".format(it, args['n_iterations'], cumulative_regret.item()))
