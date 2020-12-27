import torch


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