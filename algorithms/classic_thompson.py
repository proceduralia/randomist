import torch
from .discrete import DiscreteMabPolicyOptimization


class GaussianDiscreteThompsonSampling(DiscreteMabPolicyOptimization):
    r"""Thompson Sampling agent using Gaussian conjugate priors
        with a finite number of Gaussian policies.

    Arguments:
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        number_of_policies (int): number of discrete policies to be used (i.e., grid points)
        parameter_range ([low_bound (int), high_bound (int)]): range of each dimension of the parameters
        std_multiplier (float): multiplier for the stdv of the Gaussian prior
        policy_std (float): fixed standard deviation for all the policies
        policy_type (str): policy type among "constant", "linear"
    """
    def __init__(self, state_dim, action_dim, number_of_policies=4,
                 parameter_range=[0, 100], return_range=[-12, -1],
                 policy_std=10.0, alpha=2, policy_type="constant", setting="action_based"):
        self.return_range = return_range
        self.alpha = alpha
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         number_of_policies=number_of_policies, parameter_range=parameter_range,
                         policy_std=policy_std, policy_type=policy_type, setting=setting)

    def update_current_policy(self):
        mean_per_policy = self.arm_cumulative_payoff / self.arm_counter
        sampled_returns = torch.normal(mean_per_policy,
                                       (self.return_range[1] - self.return_range[0]) * 1 / (self.arm_counter + 1))
        self.current_policy_index = sampled_returns.argmax().item()


if __name__ == "__main__":
    algorithm = GaussianDiscreteThompsonSampling(1, 1)
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*100))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*120))
    algorithm.add_to_buffer((torch.randn(1), algorithm(torch.tensor([1])), torch.randn(1)*80))
