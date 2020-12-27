import torch
import math
from .discrete import DiscreteMabPolicyOptimization


class GIRO(DiscreteMabPolicyOptimization):
    r"""GIRO agent with a discrete number of Gaussian policies.

    Arguments:
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        number_of_policies (int): number of discrete policies to be used (i.e., grid points)
        parameter_range ([low_bound (int), high_bound (int)]): range of each dimension of the parameters
        return_range ([low_bound (int), high_bound (int)]): lower and upper bounds for the return
        policy_std (float): fixed standard deviation for all the policies
        policy_type (str): policy type among "constant", "linear"
        pseudo_rewards_per_timestep (float): how many (min, max) pseudorewards to be added.
        horizon (int): horizon of the RL problem
        setting ({'action_based', 'parameter_based'}): whether to optimize over policies or hyperpolicies
    """
    def __init__(self, state_dim, action_dim, number_of_policies=4,
                 parameter_range=[0, 100], return_range=[0, 120],
                 policy_std=10.0, policy_type="constant", pseudo_rewards_per_timestep=0.1,
                 horizon=1, setting="action_based"):
        self.pseudo_rewards_per_timestep = pseudo_rewards_per_timestep
        self.return_range = return_range
        self.horizon = horizon
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=number_of_policies, parameter_range=parameter_range,
                         policy_std=policy_std, policy_type=policy_type, setting=setting)

    def update_current_policy(self):
        sampled_expected_returns = []
        policy_indexes = torch.tensor(list(zip(*self.buffer))[0])
        returns = torch.tensor(list(zip(*self.buffer))[-1])

        for policy_idx in range(self.number_of_policies):
            # Select the returns of a given policy and initialize a uniform sampling probability
            policy_returns = returns[policy_indexes == policy_idx]
            if len(policy_returns) == 0:
                sampled_expected_returns.append(torch.tensor(float('inf')))
                continue

            # Inject exploration as prescribed by the GIRO paper
            s, a = len(policy_returns), self.pseudo_rewards_per_timestep
            as_inf, as_sup = math.floor(a*s), math.ceil(a*s)
            # Compute more or less optimistic estimate of number of pseudorewards
            n_pseudorewards = as_sup if torch.rand(1) > (a*s - as_inf) else as_inf
            pseudo_rewards = torch.cat((torch.ones(n_pseudorewards) * self.return_range[0],
                                        torch.ones(n_pseudorewards) * self.return_range[1]))
            # Insert the pseudo rewards (with max and min returns) among the real ones
            extended_returns = torch.cat((policy_returns, pseudo_rewards))
            # Pseudo rewards should be sampled with the same probability as the other returns
            sampling_weights = torch.ones_like(extended_returns)

            indices = torch.multinomial(sampling_weights, len(extended_returns),
                                        replacement=True).tolist()
            sampled_returns = extended_returns[indices]
            sampled_expected_returns.append(torch.mean(sampled_returns))

        # Select greedy and current policy
        self.current_policy_index = torch.tensor(sampled_expected_returns).argmax().item()


if __name__ == "__main__":
    algorithm = GIRO(1, 1)
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*100))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*120))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
