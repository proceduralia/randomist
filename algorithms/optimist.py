import torch
from .randomist import AbstractMediatorPO
import math
import numpy as np


class OPTIMIST(AbstractMediatorPO):
    r"""Perturbed-History Exploration agent with a discrete number of Gaussian policies.

    Arguments:
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        number_of_policies (int): number of discrete policies to be used (i.e., grid points)
        parameter_range ([low_bound (int), high_bound (int)]): range of each dimension of the parameters
        return_range ([low_bound (int), high_bound (int)]): lower and upper bounds for the return
        policy_std (float): fixed standard deviation for all the policies
        policy_type (str): policy type among "constant", "linear"
        horizon (int): horizon of the MDP
        trajectory_reuse (bool): whether to use POMITS-style trajectory reuse
        eps (float): epsilon for the Renyi
        alpha (float): hyperparameter for truncation and bias correction
        ftl (bool): whether to be optimistic or just follow the leader
        setting ({'action_based', 'parameter_based'}): whether to optimize over policies or hyperpolicies
    """
    def __init__(self, state_dim, action_dim, number_of_policies=4,
                 parameter_range=[0, 100], return_range=[0, 120],
                 policy_std=10.0, policy_type="constant",
                 horizon=1, trajectory_reuse=False, eps=1, alpha=2,
                 ftl=False, setting="action_based"):
        self.return_range = return_range
        self.trajectory_reuse = trajectory_reuse
        self.horizon = horizon
        self.eps = eps
        self.alpha = alpha
        self.ftl = ftl
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=number_of_policies, parameter_range=parameter_range,
                         policy_std=policy_std, policy_type=policy_type, setting=setting)

    def update_current_policy(self):
        if self.trajectory_reuse:
            self.update_clipped_is()
        else:
            self.update_nois()

    def update_clipped_is(self):
        returns = torch.tensor(list(zip(*self.buffer))[-1])
        weights, renyi_divergences = self.compute_importance_weights(self_normalize=False,
                                                                     renyi_alpha=1+self.eps)  # shape: (n_policies, n_returns), (n_policies)
        # Clip weights
        thresholds = ((self.t * renyi_divergences ** self.eps) / (self.alpha * math.log(self.t))) ** (1 / (1 + self.eps))
        thresholds = thresholds.view(-1, 1).expand_as(weights)
        weights = torch.where(weights > thresholds, thresholds, weights)

        mu = torch.sum(weights * returns, dim=-1)

        # Compute average returns, setting as inf the returns of unexplored arms
        if self.ftl:
            average_returns = mu
        else:
            # Positive bound
            bonus = (self.return_range[1] - self.return_range[0]) * (math.sqrt(2) + 1) * ((self.alpha * renyi_divergences * math.log(self.t))
                                                     / self.t) ** (self.eps / (1 + self.eps))
            bonus[bonus != bonus] = torch.tensor(np.inf)
            average_returns = mu + bonus

        self.current_policy_index = average_returns.argmax().item()

    def update_nois(self):
        # Select an arm that was not selected before
        if self.t < self.number_of_policies:
            # Initial round robin
            self.current_policy_index = self.t
            return
        average_returns = self.arm_cumulative_payoff / self.arm_counter
        # Select greedy and current policy
        self.current_policy_index = average_returns.argmax().item()


if __name__ == "__main__":
    algorithm = OPTIMIST(state_dim=1, action_dim=1, trajectory_reuse=True, horizon=1)
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*100))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*120))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
