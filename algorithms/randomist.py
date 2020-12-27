import torch
from torch.distributions import Normal
import warnings
from .discrete import DiscreteMabPolicyOptimization


def batched_renyi_divergence_gaussians(mu1, mu2, std1, std2, alpha=2.):
    """ Compute the Renyi divergence of order alpha between two Gaussian distributions (in closed form).
    Pairwise renyi divergences between mu1[i] and mu2[i] are computed, for diagonal Gaussians.
    See "Renyi Divergence Measures for Commonly Used Univariate Continuous Distributions"

    Arguments:
        mu1 (tensor): means of Gaussians distributions (shape: (bs, dim))
        mu2 (tensor): means of Gaussians distributions (shape: (bs, dim))
        std1 (tensor): std of Gaussians distributions (shape: (bs, dim))
        std2 (tensor): std of Gaussians distributions (shape: (bs, dim))
        alpha (float): order of the Renyi divergence
    """
    if alpha == float('inf'):
        raise NotImplementedError
    var1, var2 = std1**2, std2**2
    var_star = alpha * var2 + (1 - alpha) * var1

    term1 = alpha/2 * torch.sum((mu1 - mu2)/var_star*(mu1 - mu2), dim=-1)
    term2 = 1/(2 * (alpha - 1)) \
            * torch.log(torch.prod(var_star, dim=-1)
                       / (torch.prod(var1, dim=-1)**(1-alpha) * torch.prod(var2, dim=-1)**alpha))
    divergences = term1 - term2
    if torch.any(divergences < 0):
        warnings.warn("Negative Renyi divergence was detected!")
    divergences[divergences != divergences] = float('inf')
    return divergences


class AbstractMediatorPO(DiscreteMabPolicyOptimization):
    def __init__(self, eps=1, *args, **kwargs):
        self.eps = eps
        super().__init__(*args, **kwargs)
        # Precompute "raw" renyi divergences in closed form... Only for MABs or parameter-based settings
        assert self.setting == "parameter_based" or self.horizon == 1
        mean_batch = self.policy_params.view(len(self.policy_params), 1)
        if self.policy_type == 'my-type':
            std_batch = self.policy_std.view(len(self.policy_std), self.action_dim)
        else:
            std_batch = torch.ones_like(mean_batch) * self.policy_std
        # Compute the n^2 divergences by replicating
        m1 = mean_batch.repeat_interleave(self.number_of_policies, dim=0)
        m2 = mean_batch.repeat(self.number_of_policies, 1)
        std1 = std_batch.repeat_interleave(self.number_of_policies, dim=0)
        std2 = std_batch.repeat(self.number_of_policies, 1)
        divergences = batched_renyi_divergence_gaussians(m1, m2, std1, std2, alpha=1+self.eps)
        # Reshape into matrix form
        self.exp_divergences = divergences.reshape(self.number_of_policies, self.number_of_policies).exp()

    def normalize_weights(self, weights, keepdim=True):
        """Normalization procedure that handles corner cases"""
        if torch.sum(weights).item() == 0:
            value = 1 / len(weights)
            return torch.ones_like(weights) * value
        z = weights.sum(dim=-1, keepdim=keepdim)
        z[z != z] = 1
        z = z + 1e-16
        return weights / z

    def compute_importance_weights(self, self_normalize=False, renyi_alpha=None):
        """Compute the importance weights for each policy and for each return in the buffer.

        Arguments:
            self_normalize (bool): if True, self-normalizes the resulting importance weights
            renyi_alpha (float): if None, only the importance weights are returned. Otherwise, order of the Exp Renyi divergence

        Returns:
            importance_weights (Tensor of shape (n_policies, t)): importance weights for each policy and each collected sample
            exp_renyi_divergences (Tensor of shape (n_policies,)): harmonic mean upper bound on the Renyi of policy and mixture
        """
        # Count how many times a policy was executed for balance heuristic
        policy_counter = torch.histc(torch.tensor(list(zip(*self.buffer))[0]).float(),
                                     bins=self.number_of_policies, min=0,
                                     max=self.number_of_policies-1)
        trajectories = list(zip(*self.buffer))[1]
        # Construct state, action and reward tensors of type (n_traj, horizon, dim)
        states = torch.stack([torch.stack(trajectory[0::3])
                              for trajectory in trajectories]).view(len(trajectories), self.horizon, self.state_dim)
        actions = torch.stack([torch.stack(trajectory[1::3])
                               for trajectory in trajectories]).view(len(trajectories), self.horizon, self.action_dim)
        if self.policy_type == "constant" or self.policy_type == 'my-type':
            means_per_policy = self.policy_params.view(-1, 1, 1, self.action_dim)
            means_per_policy = means_per_policy.expand(-1, len(trajectories), self.horizon, self.action_dim)
            if self.policy_type == 'my-type':
                stds_per_policy = self.policy_std.view(-1, 1, 1, self.action_dim)
                stds_per_policy = stds_per_policy.expand(-1, len(trajectories), self.horizon, self.action_dim)
            else:
                stds_per_policy = self.policy_std
        elif self.policy_type == "linear":
            expanded_param = self.policy_params.view(self.number_of_policies, 1, 1).expand(self.number_of_policies, self.action_dim, self.state_dim)
            flattened_states = states.view(1, -1, self.state_dim).expand(self.number_of_policies, -1, -1).permute(0, 2, 1)
            means_per_policy = torch.bmm(expanded_param, flattened_states).view(self.number_of_policies,
                                                                                self.action_dim,
                                                                                len(trajectories),
                                                                                self.horizon).permute(0, 2, 3, 1)
        else:
            raise ValueError
        distr = Normal(means_per_policy, self.policy_std)
        action_logprob_per_policy = distr.log_prob(actions)  # shape: (n_policies, n_traj, horizon, action_dim)

        # Sum the logprobs for every trajectory for every policy to obtain the logprob of each trajectory under each policy
        traj_logprobs_per_policy = action_logprob_per_policy.sum(dim=(2, 3))  # shape: (n_policies, n_traj)
        log_denominators = (traj_logprobs_per_policy.exp() * policy_counter.unsqueeze(1)).sum(dim=0).log()
        log_numerators = traj_logprobs_per_policy
        importance_weights = (log_numerators - log_denominators).exp()
        if self_normalize:
            # Self-normalize importance weights
            importance_weights = self.normalize_weights(importance_weights)
        if renyi_alpha is not None:
            # Use upper bound on the renyi divergence with harmonic mean
            betas = self.arm_counter/self.t
            # Replicate weights for each policy and pre-computed divergences for each trajectory
            divergences = 1/torch.sum(betas/self.exp_divergences, dim=-1)
            return importance_weights, divergences
        else:
            return importance_weights
