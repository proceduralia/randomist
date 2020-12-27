import torch
from .randomist import AbstractMediatorPO
from torch.distributions import Binomial
import math


class PHE(AbstractMediatorPO):
    r"""Perturbed-History Exploration agent with a discrete number of Gaussian policies.

    Arguments:
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        number_of_policies (int): number of discrete policies to be used (i.e., grid points)
        parameter_range ([low_bound (int), high_bound (int)]): range of each dimension of the parameters
        return_range ([low_bound (int), high_bound (int)]): lower and upper bounds for the return
        policy_std (float): fixed standard deviation for all the policies
        policy_type (str): policy type among "constant", "linear"
        pseudo_rewards_per_timestep (float): how many (min, max) pseudorewards to be added.
        horizon (int): horizon of the MDP
        mode ({'no_is', 'randomist'}): mode for the algorithm
        eps (float): \epsilon value for Renyi divergence
        alpha (float): coefficient for bias correction and truncation
        setting ({'action_based', 'parameter_based'}): whether to optimize over policies or hyperpolicies
    """
    def __init__(self, state_dim, action_dim, number_of_policies=4,
                 parameter_range=[-5, 5], return_range=[-12, -1],
                 policy_std=1.0, policy_type="constant",
                 pseudo_rewards_per_timestep=1.1, horizon=1,
                 mode='no_is', eps=1, alpha=2, setting="action_based"):
        self.pseudo_rewards_per_timestep = pseudo_rewards_per_timestep
        self.return_range = return_range
        assert mode in {'no_is', 'randomist'}
        self.mode = mode
        self.alpha = alpha
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=number_of_policies, parameter_range=parameter_range,
                         policy_std=policy_std, policy_type=policy_type, eps=eps, setting=setting,
                         horizon=horizon)

    def update_current_policy(self):
        if self.mode == 'randomist':
            self.update_randomist()
        elif self.mode == 'no_is':
            self.update_nois()
        else:
            raise ValueError

    def update_randomist(self):
        returns = torch.tensor(list(zip(*self.buffer))[-1])
        importance_weights, exp_renyi_divergences = self.compute_importance_weights(self_normalize=False,
                                                                                    renyi_alpha=1+self.eps)  # shape: (n_policies, n_returns), (n_policies,)
        ess = self.t / (exp_renyi_divergences)
        ns_additional_rewards = torch.ceil(ess * self.pseudo_rewards_per_timestep)

        distr = Binomial(ns_additional_rewards, torch.ones_like(ns_additional_rewards)*0.5)
        sampled_sums_01 = distr.sample() / ns_additional_rewards

        # Normalize the samples from the binomial distribution into the appropriate range
        min_s, max_s = self.return_range[0] * ns_additional_rewards, self.return_range[1] * ns_additional_rewards
        U = sampled_sums_01 * (max_s - min_s) + min_s
        eta = U / ns_additional_rewards
        bias_correction = torch.sqrt((self.alpha*math.log(self.t))/ess)
        eta = eta + bias_correction

        # Clip weights
        thresholds = torch.sqrt((exp_renyi_divergences * self.t) / (self.alpha*math.log(self.t)))
        thresholds = thresholds.view(-1, 1).expand_as(importance_weights)
        importance_weights = torch.where(importance_weights > thresholds, thresholds, importance_weights)

        # MIST estimator
        mu = torch.sum(importance_weights * returns, dim=-1)

        bias_correction = (self.return_range[1] - self.return_range[0]) * ((self.alpha * exp_renyi_divergences * math.log(self.t))
                                                    / self.t) ** (self.eps / (1 + self.eps))
        # Compute average returns
        average_returns = mu + eta + bias_correction

        # Select current policy
        self.current_policy_index = average_returns.argmax().item()

    def update_nois(self):
        ns_additional_rewards = torch.ceil(self.arm_counter * self.pseudo_rewards_per_timestep)
        distr = Binomial(ns_additional_rewards, torch.ones_like(ns_additional_rewards)*0.5)
        sampled_sums_01 = distr.sample() / ns_additional_rewards

        # Normalize the samples from the binomial distribution into the appropriate range
        min_s, max_s = self.return_range[0] * ns_additional_rewards, self.return_range[1] * ns_additional_rewards
        sampled_sums = sampled_sums_01 * (max_s - min_s) + min_s

        denominator = self.arm_counter + ns_additional_rewards
        numerator = self.arm_cumulative_payoff + sampled_sums
        average_returns = numerator / denominator

        # Select current policy
        self.current_policy_index = average_returns.argmax().item()


if __name__ == "__main__":
    algorithm = PHE(state_dim=1, action_dim=1, mode="clipped_is", horizon=1, setting="parameter_based")
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.randn(1)), torch.randn(1)*100))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.randn(1)), torch.randn(1)*120))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.randn(1)), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.randn(1)), torch.randn(1)*80))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.randn(1)), torch.randn(1)*80))
