import gym
import torch


def collect_trajectory(environment, agent, horizon=None):
    r"""Collect a trajectory using an agent in an environment for a number of steps.

    Arguments:
        environment (gym env): environment following the OpenAI Gym API
        agent (callable): object that returns an action given a state
        horizon (int): number of timesteps per episode. If None, the episode is finished when done
    Returns:
        trajectory (tuple): trajectory of the type [s0, a0, r1, ..., sn]
    """
    done = False
    trajectory = []
    state = environment.reset().unsqueeze(0)
    for t in range(horizon):
        action = agent(state, greedy=False)
        next_state, reward, done, _ = environment.step(action)
        trajectory.extend([state, action, reward])
        state = next_state
        if done:
            break
    return tuple(trajectory)


def trajectory_to_tensors(trajectory):
    r"""Convert a trajectory to state, action and reward tensors.

    Arguments:
        trajectory (tuple): trajectory of the type [s0, a0, r1, ..., sn]
    Returns:
        states (tensor): tensor of shape (horizon, dim_state)
        actions (tensor): tensor of shape (horizon, dim_action)
        rewards (tensor): tensor of shape (horizon,)
    """
    states = torch.stack(trajectory[0::3])
    actions = torch.stack(trajectory[1::3])
    rewards = torch.tensor(trajectory[2::3])
    return states, actions, rewards


def compute_J(environment, agent, n_episodes, horizon=None):
    empirical_returns = []
    for ev in range(n_episodes):
        trajectory = collect_trajectory(environment=environment, agent=agent,
                                        horizon=horizon)
        J = torch.sum(torch.tensor(trajectory[2::3])).item()
        empirical_returns.append(J)
    average_return = torch.tensor(empirical_returns).mean()
    return average_return


class TorchTensorWrapper(gym.ActionWrapper):
    """Wraps the environment to work with torch tensors instead of numpy arrays"""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        action = action.detach().numpy()
        obs, reward, done, info = self.env.step(action)
        return torch.tensor(obs, dtype=torch.float), reward, done, info

    def reset(self):
        return torch.tensor(self.env.reset(), dtype=torch.float)
