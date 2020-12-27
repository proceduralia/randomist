from gym.envs.registration import register

register(
    id='LQG1D-v0',
    entry_point='envs.lqg1d:LQG1D'
)

register(
    id='ContinuousCartpole-v0',
    entry_point='envs.cartpole:ContCartPole'
)
