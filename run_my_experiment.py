import torch
import numpy as np
import random
import argparse
from mellog.logger import Mellogger
from envs.mab import MyMAB
from algorithms.classic_thompson import GaussianDiscreteThompsonSampling
from algorithms.giro import GIRO
from algorithms.phe import PHE
from algorithms.optimist import OPTIMIST
from algorithms.ucb1 import UCB1
from algorithms.gpucb import GPUCB
from scipy.special import erf


def compute_arm_value(m, s):
    return 1 / 8 * (4 + (-np.exp(-((-4 + m) ** 2) / (2 * s ** 4)) + np.exp(-(m ** 2) / (2 * s ** 4))) * np.sqrt(2 / np.pi) * s ** 2
                    - (-4 + m) * erf((-4 + m) / (np.sqrt(2) * s ** 2)) + m * erf(m / (np.sqrt(2) * s ** 2)))


parser = argparse.ArgumentParser(description="Script for running a mab experiment.")
# Parameters for any algorithm
parser.add_argument('--n_iterations', type=int, default=10000, help="Number of iterations")
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
parser.add_argument('--sigma', type=float, default=0.9, help="Std for Gaussian policies")
parser.add_argument('--lambda', type=float, default=0.9, help="Std for Gaussian policies")
parser.add_argument('--pseudo_rewards_per_timestep', type=float, default=1.1,
                    help="""Number of pseudorewards for unit of history in GIRO, PHE and RANDOMIST.""")
args = vars(parser.parse_args())

# Seed pseudo-random number generators
torch.manual_seed(args['random_seed']), np.random.seed(args['random_seed']), random.seed(args['random_seed'])
# Initialize logger
logger = Mellogger(log_dir=args['logdir'], exp_name=args['exp_name'], args=args, test_mode=False)

# Initialize environment
mab = MyMAB()
state_dim = 1
action_dim = 1

sigma_ = args['sigma']
lambda_ = args['lambda']

stds = [sigma_, lambda_]

# Initialize algorithm
if args['algorithm'] == "gaussian":
    algorithm = GaussianDiscreteThompsonSampling(state_dim=state_dim, action_dim=action_dim,
                                                 policy_std=stds, policy_type="my-type",
                                                 return_range=[0, 1])
elif args['algorithm'] == "giro":
    algorithm = GIRO(state_dim=state_dim, action_dim=action_dim,
                     policy_std=stds, policy_type="my-type",
                     pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                     return_range=[0, 1], horizon=1)
elif args['algorithm'] == "phe":
    algorithm = PHE(state_dim=state_dim, action_dim=action_dim,
                    policy_std=stds, policy_type="my-type",
                    pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                    return_range=[0, 1],
                    horizon=1, mode='no_is', setting='action_based')
elif args['algorithm'] == "randomist":
    algorithm = PHE(state_dim=state_dim, action_dim=action_dim,
                    policy_std=stds, policy_type="my-type",
                    pseudo_rewards_per_timestep=args['pseudo_rewards_per_timestep'],
                    return_range=[0, 1],
                    horizon=1, mode='randomist', setting='action_based')
elif args['algorithm'] == "optimist":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         policy_std=stds, policy_type="my-type",
                         return_range=[0, 1],
                         horizon=1, trajectory_reuse=True, ftl=False)
elif args['algorithm'] == "ucb1":
    algorithm = UCB1(state_dim=state_dim, action_dim=action_dim,
                     policy_std=stds, policy_type="my-type",
                     return_range=[0, 1],
                     horizon=1)
elif args['algorithm'] == "gpucb":
    algorithm = GPUCB(state_dim=state_dim, action_dim=action_dim,
                      policy_std=stds, policy_type="my-type",
                      return_range=[0, 1],
                      horizon=1)
elif args['algorithm'] == "ftl":
    algorithm = OPTIMIST(state_dim=state_dim, action_dim=action_dim,
                         policy_std=stds, policy_type="my-type",
                         return_range=[0, 1],
                         horizon=1, trajectory_reuse=True, ftl=True)
else:
    raise ValueError(args['algorithm']+"is not a valid algorithm!")

dummy_state = torch.randn(1)
training_rewards = []
cumulative_regret = 0

arm_values = [compute_arm_value(m, s) for m, s in zip(algorithm.policy_params, algorithm.policy_std)]
optimal_arm_value = max(arm_values)

for it in range(args['n_iterations']):
    # Collect a new interaction
    action = algorithm(dummy_state, greedy=False)
    reward = mab(action)
    training_rewards.append(reward)

    # Add interaction to buffer, triggering update of policy
    trajectory = [dummy_state, action, reward]
    algorithm.add_to_buffer(trajectory)

    if it % args['evaluate_every'] == 0:
        # Compute regret
        action = algorithm(dummy_state, greedy=False, evaluation_mode=True)
        index = np.asscalar(np.where(action == algorithm.policy_params)[0])

        instantaneous_regret = optimal_arm_value - arm_values[index]
        cumulative_regret = cumulative_regret + instantaneous_regret
        logger.log("policy_index", algorithm.current_policy_index)
        logger.log("evaluation_actions", action.numpy())
        logger.log("evaluation_rewards", reward.item())
        logger.log("cumulative_regret", cumulative_regret.item())
    if args['logging_freq'] is not None and it % args['logging_freq'] == 0:
        print("[{}/{}] Regret={}".format(it, args['n_iterations'], cumulative_regret.item()))
