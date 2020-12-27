import torch
from .discrete import DiscreteMabPolicyOptimization
import math


class UCB1(DiscreteMabPolicyOptimization):
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
        alpha (float): hyperparameter for exploration bonus
    """
    def __init__(self, state_dim, action_dim, number_of_policies=4,
                 parameter_range=[0, 100], return_range=[0, 120],
                 policy_std=10.0, policy_type="constant",
                 horizon=1, alpha=2,
                 setting="action_based"):
        self.return_range = return_range
        self.horizon = horizon
        self.alpha = alpha
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=number_of_policies, parameter_range=parameter_range,
                         policy_std=policy_std, policy_type=policy_type, setting=setting)

    def update_current_policy(self):
        # Select an arm that was not selected before
        if self.t < self.number_of_policies:
            # Initial round robin
            self.current_policy_index = self.t
            return
        bonus = (self.return_range[1] - self.return_range[0]) * (self.alpha * math.log(self.t) / (2 * self.arm_counter)) ** 0.5
        average_returns = self.arm_cumulative_payoff / self.arm_counter + bonus
        # Select greedy and current policy
        self.current_policy_index = average_returns.argmax().item()


if __name__ == "__main__":
    algorithm = UCB1(state_dim=1, action_dim=1, horizon=1)
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*100))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*120))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
