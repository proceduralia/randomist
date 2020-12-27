from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.stats import binom
import math
from functools import lru_cache, partial
import numpy as np
from .randomist import batched_renyi_divergence_gaussians

# Activate memoization on binomial pmf and cdf
binomial_logcdf = lru_cache(maxsize=10000)(binom.logcdf)
binomial_pmf = lru_cache(maxsize=10000)(binom.pmf)
binomial_cdf = lru_cache(maxsize=10000)(binom.cdf)


class RandomistMCMC:
    """RANDOMIST with MCMC sampling from posterior distribution of the maximum.
    Parameter-based with Gaussian hyperpolicies and linear policies.
    """
    Episode = namedtuple('Episode', ['policy_params', 'trajectory', 'cumulative_payoff'])

    def __init__(self, state_dim, action_dim, parameter_range=torch.tensor([[-5, 5]]), policy_std=torch.tensor([0.15, 3]),
                 alpha=2, eps=1, a=1.1, nMCMCsteps=10, use_bias=False, return_range=[-95, 95]):
        self.parameter_range = parameter_range
        self.policy_std = policy_std
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.a = a
        self.eps = eps
        self.nMCMCsteps = nMCMCsteps
        self.use_bias = use_bias
        self.return_range = return_range

        # Initialize the list of former hyperpolicy parameters
        self.hyperparams_list = []
        # Randomly initialize hyperpolicy
        self.hyperpolicy_dim = state_dim*action_dim+action_dim if use_bias else state_dim*action_dim
        self.hyperpolicy_params = torch.zeros(self.hyperpolicy_dim)
        self.hyperparams_list.append(self.hyperpolicy_params)
        # Sample a policy to be used
        self.sample_from_hyperpolicy()

        # Buffer retaining episode named tuples
        self.buffer = []
        self.t = 0

        self.log_denominators = torch.zeros(0)  # (t-1,) vector
        self.log_probabilities = torch.empty(0, 0)  # (t-1, t-1) matrix of the type (number_of_policies, number_of_hyperpolicies)
        self.exp_divergences_inner_sums = torch.empty(0)  # (t-1,) vector with the sum of the exp divergences inside the harmonic mean

    def __call__(self, state, greedy=False, evaluation_mode=False):
        # Use the sampled (linear) policy
        state = state.unsqueeze(0) if len(state.size()) == 1 else state
        actions = F.linear(state, self.weight, self.bias if self.use_bias else torch.zeros(self.action_dim))
        return actions.view(actions.size(1)) if len(state.size()) == 2 else actions

    def clip_params(self, params):
        params = torch.where(params < self.parameter_range[:, 0],
                             self.parameter_range[:, 0], params)
        params = torch.where(params > self.parameter_range[:, 1],
                             self.parameter_range[:, 1], params)
        return params

    def update_current_policy(self):
        """Sample a new hyperpolicy.
        Called right after the insertion of a new trajectory in the buffer.
        """
        self.t += 1
        # Cache approach: save the sum of the probs up to time t-1 for each of the t-1 policies
        policy_params = torch.stack(list(zip(*self.buffer))[0])
        means_per_policy = torch.stack([params for params in self.hyperparams_list], dim=0)  # (t, policy_param_dim)
        # We will compute two types probabilities: of the last policy under all hyperpolicies (t-1,),
        # and of all the policies under the last hyperpolicy (t,)

        # Compute probability of last policy under old hyperpolicies
        distr = Normal(means_per_policy[:-1], self.policy_std)
        new_logprobs_per_old_hpolicy = distr.log_prob(policy_params[-1])  # shape: (t-1, policy_param_dim)
        # Probability of last policy under every hyperpolicy
        new_logprobs_per_old_hpolicy = new_logprobs_per_old_hpolicy.sum(dim=-1)  # shape: (t-1)

        # Compute probability of all policies under last hyperpolicy
        distr = Normal(means_per_policy[-1], self.policy_std)
        logprobs_per_new_hpolicy = distr.log_prob(policy_params)  # shape: (t, policy_param_dim)
        logprobs_per_new_hpolicy = logprobs_per_new_hpolicy.sum(dim=-1)  # shape: (t,)

        # For t-1 denominators, I can exploit caching
        old_log_denominators = (self.log_denominators.exp() + logprobs_per_new_hpolicy[:self.t-1].exp()).log()
        # For the last denominator, add the probabilities of last policy under old hyperpolicies to probability of last policy under last hyperpolicy
        last_log_denominator = (new_logprobs_per_old_hpolicy.exp().sum() + logprobs_per_new_hpolicy[-1].exp()).log()
        log_denominators = torch.cat((old_log_denominators, last_log_denominator.reshape(1)), dim=0)
        self.log_denominators = log_denominators
        # Update probabilities cache. First add the row referring to the last trajectory for old hyperpolicies
        self.log_probabilities = torch.cat((self.log_probabilities, new_logprobs_per_old_hpolicy.view(1, self.t-1)),
                                           dim=0)  # shape: (t, t-1)
        # Then add the column with the probabilities under the last hyperpolicy
        self.log_probabilities = torch.cat((self.log_probabilities, logprobs_per_new_hpolicy.view(self.t, 1)),
                                           dim=1)  # shape: (t, t)

        # Compute divergences of each hyperpolicy w.r.t. each other
        # Only compute divergences between last hyperpolicy and all the other ones
        m1 = means_per_policy[-1].unsqueeze(0).repeat(self.t, 1)  # Repeat last policy
        m2 = means_per_policy
        std = self.policy_std
        divergences = batched_renyi_divergence_gaussians(m2, m1, std, std, alpha=1+self.eps)        # shape: (t,)
        self.exp_divergences_inner_sums = torch.cat((self.exp_divergences_inner_sums,
                                                     torch.zeros(1)), dim=0) + 1/divergences.exp()  # shape: (t,)
        # Use computed inner sums to obtain the harmonic mean of the exponentiated divergences
        exp_divergences = self.t/self.exp_divergences_inner_sums

        # Compute ess for each executed policy
        other_esss = self.t/exp_divergences

        # Compute mu+bias for each executed policy
        returns = torch.tensor(list(zip(*self.buffer))[-1])
        importance_weights = (self.log_probabilities.t() - log_denominators).exp()  # shape: (t, t)
        # Clip weights
        threshold = torch.sqrt((exp_divergences * self.t) / (self.alpha*math.log(self.t)))
        thresholds = threshold.expand_as(importance_weights)
        importance_weights = torch.where(importance_weights > thresholds, thresholds, importance_weights)
        other_mus = torch.sum(importance_weights * returns, dim=-1)
        # Add bias term
        other_mus = other_mus + torch.sqrt((self.alpha*math.log(self.t)/other_esss))

        # Select the new hyperpolicy by taking a number of MCMC steps
        self.hyperpolicy_params = self.MCMCwalk(policy_params, means_per_policy, log_denominators,
                                                other_esss, other_mus)

    def MCMCwalk(self, policy_params, means_per_policy, log_denominators, other_esss, other_mus):
        args = [policy_params, means_per_policy, log_denominators, other_esss, other_mus]
        return self.mh_sampling(means_per_policy[-1], args=args)

    def mh_sampling(self, initial_point, args=None):
        # Args are the fixed parameters for the log density

        # By default, use the std of the hyperpolicy for the kernel
        current_point = initial_point
        current_logdensity = self.max_posterior_log_density(current_point, *args)
        for _ in range(self.nMCMCsteps):
            # Sample a new proposal point
            unclipped_proposed_point = torch.normal(current_point, self.policy_std)
            proposed_point = self.clip_params(unclipped_proposed_point)

            # Compute the logdensity for the proposed point
            proposed_logdensity = self.max_posterior_log_density(proposed_point, *args)

            ratio = math.exp(proposed_logdensity - current_logdensity)

            if torch.rand(1).item() <= ratio:
                current_point = proposed_point
                current_logdensity = proposed_logdensity
        return current_point

    def add_to_buffer(self, trajectory):
        """Adds a trajectory to buffer. Assume the trajectory was collected by the current policy.

        A trajectory is a list of the type [s0, a0, r0, s1, a1, r1]
        """
        J = torch.sum(torch.tensor(trajectory[2::3]))
        parameter_vector = torch.cat((self.weight.flatten(), self.bias), dim=0) if self.use_bias else self.weight.flatten()
        episode = type(self).Episode(parameter_vector, trajectory, J)
        self.buffer.append(episode)

        # Update hyperpolicy and policy
        self.update_current_policy()
        self.hyperparams_list.append(self.hyperpolicy_params)
        self.sample_from_hyperpolicy()

    def sample_from_hyperpolicy(self):
        # Sample from current hyperpolicy to update current policy
        flattened_params = torch.normal(self.hyperpolicy_params, self.policy_std)  # shape: (state_dim*action_dim+action_dim)
        self.weight = flattened_params[:self.state_dim*self.action_dim].reshape(self.action_dim, self.state_dim)
        if self.use_bias:
            self.bias = flattened_params[-self.action_dim:]

    def max_posterior_log_density(self, theta, policy_params, means_per_policy, log_denominators, other_esss, other_mus):
        returns = torch.tensor(list(zip(*self.buffer))[-1])
        # Compute log density of the posterior distribution of the maximum
        distr = Normal(theta, self.policy_std)
        log_numerators = distr.log_prob(policy_params).sum(dim=-1)  # shape: (t,)
        importance_weights = (log_numerators - log_denominators).exp()
        # Compute renyi divs between theta and the other distributions
        std = self.policy_std
        exp_divergences = batched_renyi_divergence_gaussians(theta.unsqueeze(0).repeat(self.t, 1), means_per_policy,
                                                             std, std, alpha=1+self.eps).exp()  # shape: (t,)
        exp_divergence = 1/torch.mean(1/exp_divergences)
        ess = self.t / exp_divergence
        if ess.item() == 0:
            return -float('inf')

        # Clip weights
        threshold = torch.sqrt((exp_divergence * self.t) / (self.alpha*math.log(self.t)))
        thresholds = threshold.expand_as(importance_weights)
        importance_weights = torch.where(importance_weights > thresholds, thresholds, importance_weights)
        mu = torch.sum(importance_weights * returns, dim=-1) + torch.sqrt((self.alpha*math.log(self.t)/ess))

        # Compute argument of the cdf for each old arm and for each i
        i_ = torch.arange(torch.ceil(self.a*ess).item() + 1)
        i_s = i_.view(-1, 1).repeat(1, self.t)
        min_s, max_s = self.return_range[0], self.return_range[1]
        cdf_argument = (i_s*other_esss/(ess+1e-16) + other_esss*(mu - other_mus)/(max_s-min_s))  # shape: (len(i_s), t)
        cdf_argument = torch.floor(cdf_argument)

        # For each row of the cdf_argument, the cdf parameters are exactly the same
        # Apply the vectorized cdf to each row that share these parameters
        # This should be faster than simply calling the binomial.logcdf with an array of different esss,
        # because it can leverage memoization
        logcdf_values = []
        for t, arguments in enumerate(cdf_argument.transpose(1, 0)):
            # Arguments is a (len(i_s),) containing the arguments for the cdf for a fixed t
            logcdf = np.vectorize(partial(binomial_logcdf, n=torch.ceil(self.a*other_esss[t]).item(), p=0.5))
            # Apply vectorized memoized logcdf to all the arguments for a fixed t
            logcdf_for_a_t = torch.tensor(logcdf(arguments.numpy()))  # shape: (len(i_s))
            logcdf_values.append(logcdf_for_a_t)
        logcdf_values = torch.stack(logcdf_values).transpose(1, 0)  # shape: (len(i_s), t)
        # Compute the product of the cdfs for a given i
        accumulated_cdfs = torch.exp(torch.sum(logcdf_values, dim=-1))  # shape: (len(i_s),)

        # Compute pmf by vectorizing and applying pmf with parameters determined by ESS of current particle
        pmf = np.vectorize(partial(binomial_pmf, n=torch.ceil(self.a*ess).item(), p=0.5))
        evaluated_pmfs = torch.tensor(pmf(i_.numpy()))  # shape: (len(i_))

        cdf = np.vectorize(partial(binomial_cdf, n=torch.ceil(self.a * ess).item(), p=0.5))
        evaluated_cdfs = torch.tensor(cdf(i_.numpy()))  # shape: (len(i_))

        log_density = torch.sum(evaluated_pmfs / evaluated_cdfs * accumulated_cdfs).log().item()
        return log_density


if __name__ == "__main__":
    s_dim = 2
    a_dim = 1
    algorithm = RandomistMCMC(state_dim=s_dim, action_dim=a_dim)
    for i in range(2):
        algorithm.add_to_buffer((torch.randn(s_dim), algorithm(torch.randn(s_dim)), torch.randn(1)))
