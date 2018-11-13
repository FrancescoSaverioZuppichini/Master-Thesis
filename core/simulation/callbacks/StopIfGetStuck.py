import numpy as np

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException


class StopIfGetStuck(SimulationCallback):
    def __init__(self, n_last, tol=0.1):
        self.history = []
        self.n_last = n_last
        self.tol = tol

    def tick(self, sim, world, agent, *args, **kwargs):
        position = agent.state['pose'].pose.position

        if len(self.history) >= self.n_last: self.history.pop(0)

        self.history.append([position.x, position.y])

        std = np.std(self.history)
        
        if (std < self.tol).any(): raise SimulationException
