from torch.distributions import Normal
from collections import namedtuple
import torch
import random
import abc
import numpy as np


class DiscreteMabPolicyOptimization(metaclass=abc.ABCMeta):
    Episode = namedtuple('Episode', ['policy_index', 'trajectory', 'cumulative_payoff'])

    def __init__(self, state_dim, action_dim, number_of_policies=10,
                 parameter_range=[0, 100], policy_std=10.0, policy_type="constant",
                 setting="action_based", horizon=1):
        self.number_of_policies = number_of_policies
        self.policy_type = policy_type
        self.parameter_range = parameter_range
        self.policy_std = policy_std
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        assert setting in {"action_based", "parameter_based"}
        self.setting = setting
        assert policy_type in {"constant", "linear", "my-type"}
        if setting == "action_based":
            if policy_type == "constant":
                n_points = int(np.ceil(number_of_policies ** (1 / action_dim)))
                coordinates = torch.linspace(parameter_range[0], parameter_range[1], n_points)
                meshgrid = torch.meshgrid(*([coordinates]*action_dim))
                self.policy_params = torch.tensor(list(coord.flatten().tolist() for coord in meshgrid)).t()
            elif policy_type == "linear":
                # Linear policy with parameter sharing over the states
                coordinates = torch.linspace(parameter_range[0], parameter_range[1], number_of_policies)
                self.policy_params = torch.tensor(list(coord.flatten().tolist() for coord in coordinates))
            elif policy_type == 'my-type':
                self.policy_params = torch.tensor([0., 1., 2., 2.95, 3.])
                self.number_of_policies = 5
                sigma_, lambda_ = self.policy_std
                self.policy_std = torch.tensor([sigma_, sigma_, sigma_, lambda_, sigma_])
        elif setting == "parameter_based":
            # Only considering linear policies
            param_dim = state_dim * action_dim
            n_points = number_of_policies // param_dim
            coordinates = torch.linspace(parameter_range[0], parameter_range[1], n_points)
            meshgrid = torch.meshgrid(*([coordinates]*param_dim))
            self.policy_params = torch.tensor(list(coord.flatten().tolist() for coord in meshgrid)).t()
        # Buffer retaining episode named tuples
        self.buffer = []
        # Index in the list of policies of the current policy, initially random
        self.current_policy_index = random.randint(0, number_of_policies-1)
        if setting == "parameter_based":
            self.sample_from_hyperpolicy()
        # Initialize arm counter and cumulative payoff memory
        self.arm_counter = torch.zeros(self.number_of_policies)
        self.arm_cumulative_payoff = torch.zeros(self.number_of_policies)
        self.t = 0

    def __call__(self, state, greedy=False, evaluation_mode=False):
        index = self.current_policy_index
        if not evaluation_mode:
            self.arm_counter[index] = self.arm_counter[index] + 1
            self.t = self.t + 1

        if self.setting == "action_based":
            if self.policy_type == "constant":
                actions = self.policy_params[index].repeat(len(state))
            elif self.policy_type == "linear":
                expanded_param = self.policy_params[index].expand(self.action_dim, self.state_dim)
                actions = expanded_param.matmul(state.t()).t()
            elif self.policy_type == 'my-type':
                actions = self.policy_params[index].repeat(len(state))
            else:
                raise ValueError("Invalid policy type")
            if not evaluation_mode:
                if self.policy_type == 'my-type':
                    stds = self.policy_std[index].repeat(len(state))
                else:
                    stds = self.policy_std
                actions = Normal(actions, stds).sample()
        elif self.setting == "parameter_based":
            # Use the sampled (linear) policy
            actions = self.sampled_params.matmul(state.t()).t()
        else:
            raise ValueError
        return actions

    @abc.abstractmethod
    def update_current_policy(self):
        """Picks a new policy from the set of the available one.
        Called right after the insertion of a new trajectory in the buffer.
        The method should modify 'self.current_policy_index' and
        'self.greedy_policy_index'.
        """

    def add_to_buffer(self, trajectory):
        """Adds a trajectory to buffer. Assume the trajectory was collected by the current policy.

        A trajectory is a list of the type [s0, a0, r0, s1, a1, r1]
        """
        J = torch.sum(torch.tensor(trajectory[2::3]))
        self.arm_cumulative_payoff[self.current_policy_index] = self.arm_cumulative_payoff[self.current_policy_index] + J
        episode = type(self).Episode(self.current_policy_index, trajectory, J)
        self.buffer.append(episode)
        # Update either hyperpolicy or policy
        self.update_current_policy()
        if self.setting == "parameter_based":
            self.sample_from_hyperpolicy()

    def sample_from_hyperpolicy(self):
        # Sample from current hyperpolicy to update current policy
        flattened_params = Normal(self.policy_params[self.current_policy_index], self.policy_std).sample()  # shape: (state_dim*action_dim)
        self.sampled_params = flattened_params.reshape(self.state_dim, self.action_dim)