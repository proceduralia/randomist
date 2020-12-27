import torch
import math
import numpy as np
from .lqg1d import LQG1D

class AckleyContinuousMAB:
    def __init__(self, noise_magnitude=0.5):
        self.noise_magnitude = noise_magnitude
        self.optimal_value = -self.ackley_function(torch.tensor(0.0))

    def ackley_function(self, action):
        arg1 = -0.2 * torch.sqrt(0.5 * (action ** 2))
        arg2 = 0.5 * (torch.cos(2. * math.pi * action))
        ackley_value = (-20. * torch.exp(arg1) - torch.exp(arg2) + 20. + math.e)
        return ackley_value

    def make_deterministic(self):
        self.former_noise = self.noise_magnitude
        self.noise_magnitude = 0

    def make_stochastic(self):
        self.noise_magnitude = self.former_noise

    def optimal_arm_value(self, n_arms):
        # Discretize
        arm_means = torch.linspace(-5, 5, n_arms)
        function_values = -self.ackley_function(arm_means)
        return function_values.max()

    def __call__(self, action):
        ackley_value = self.ackley_function(action)
        return -ackley_value + torch.randn_like(ackley_value) * self.noise_magnitude

class RastriginContinuousMAB:
    def __init__(self, noise_magnitude=0.5):
        self.noise_magnitude = noise_magnitude
        self.optimal_value = -self.rastrigin_function(torch.tensor(0.0))

    def rastrigin_function(self, action):
        return 10 + action ** 2 - 10 * torch.cos(2 * math.pi * action)

    def make_deterministic(self):
        self.former_noise = self.noise_magnitude
        self.noise_magnitude = 0

    def make_stochastic(self):
        self.noise_magnitude = self.former_noise

    def optimal_arm_value(self, n_arms):
        # Discretize
        arm_means = torch.linspace(-5.12, 5.12, n_arms)
        function_values = -self.rastrigin_function(arm_means)
        return function_values.max()

    def __call__(self, action):
        rastrigin_value = self.rastrigin_function(action)
        return -rastrigin_value + torch.randn_like(rastrigin_value) * self.noise_magnitude

class LeonContinuousMAB:
    def __init__(self, noise_magnitude=0.5):
        self.noise_magnitude = noise_magnitude
        self.optimal_value = -self.leon_function(torch.tensor([1., 1.]))

    def leon_function(self, action):
        return 100 * (action[1] - action[0] ** 3) ** 2 + (1 - action[0]) ** 2

    def make_deterministic(self):
        self.former_noise = self.noise_magnitude
        self.noise_magnitude = 0

    def make_stochastic(self):
        self.noise_magnitude = self.former_noise

    def optimal_arm_value(self, n_arms):
        # Discretize
        n_arms = int(np.ceil(n_arms ** (1 / 2)))
        arm_means = torch.linspace(-2, 2, n_arms)
        meshgrid = torch.meshgrid(arm_means, arm_means)
        arms = torch.tensor(list(coord.flatten().tolist() for coord in meshgrid)).t()
        #print(arms)
        function_values = torch.tensor([-self.leon_function(a) for a in arms])
        return function_values.max()

    def __call__(self, action):
        leon_value = self.leon_function(action)
        return -leon_value + torch.randn_like(leon_value) * self.noise_magnitude

class MyMAB:
    def __init__(self):
        self.noise_magnitude = 0.

    def function(self, action):
        return torch.clamp(action / 4, min=0, max=1)

    def make_deterministic(self):
        self.former_noise = self.noise_magnitude
        self.noise_magnitude = 0

    def make_stochastic(self):
        self.noise_magnitude = self.former_noise

    def __call__(self, action):
        value = self.function(action)
        return value

class LQGMAB:
    def __init__(self):
        self.mdp = LQG1D()
        self.gamma = self.mdp.gamma
        self.horizon = self.mdp.horizon
        self.noise_magnitude = 0.

    def function(self, action):
        gain = action
        _ret = 0
        t = 0
        done = False
        s = self.mdp.reset()
        while not done and t < self.horizon:
            a = gain * s
            s, r, done, _ = self.mdp.step(a)
            _ret = _ret + r * self.gamma ** t
            t += 1

        _ret /= (1 - self.gamma**(self.horizon)) / (1 - self.gamma)
        return torch.tensor(_ret)

    def make_deterministic(self):
        self.former_noise = self.noise_magnitude
        self.noise_magnitude = 0

    def make_stochastic(self):
        self.noise_magnitude = self.former_noise

    def __call__(self, action):
        value = self.function(action)
        return value


class HarderContinuousMAB:
    def __init__(self, noise_magnitude=30, clip=True):
        self.noise_magnitude = noise_magnitude

    def __call__(self, action):
        return 100 * torch.sin(action/20) + 100 * torch.sin(action) + \
               action + torch.randn_like(action) * self.noise_magnitude

    def random_return(self):
        return torch.rand() * 100


class EasyContinuousMAB:
    def __init__(self, noise_magnitude=5):
        self.noise_magnitude = noise_magnitude

    def __call__(self, action):
        return 100 * torch.sin(action/20) + action + torch.randn_like(action) * self.noise_magnitude


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    n_points = 100
    mab = RastriginContinuousMAB(noise_magnitude=0)
    coordinates = torch.linspace(-5.12, 5.12, n_points)
    values = []
    s = 0.5
    for x in coordinates:
        v = []
        for i in range(100):
            action = x + np.random.randn() * s
            v.append(mab(action))
        values.append(v)

    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1)
    plt.plot(coordinates, means, marker='*')
    plt.fill_between(coordinates, means+stds, means-stds, alpha=0.2)
    plt.plot(coordinates, torch.ones_like(coordinates) * mab.optimal_arm_value(n_points), marker='*')
    plt.show()
    print(mab.optimal_arm_value(n_points))
