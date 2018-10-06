import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException
# TODO this is a naive implementation, a more robust approach
# should be tracing the z and check how much it changed in time
class OutOfMap(SimulationCallback):
    def __init__(self, x, y, z=0, tol=-.01):
        self.z = z
        self.x = x
        self.y = y
        self.tol = tol

    def tick(self, sim, world, agent, *args, **kwargs):
        pose = agent.state['pose']
        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z

        exp = None

        if isinstance(self.x, tuple):
            # lower bound
            if x - self.tol <= self.x[0]:
                exp = SimulationException('Agent is going to fall!')
            # upper bound
            elif x + self.tol >= self.x[1]:
                exp = SimulationException('Agent is going to fall!')

        if isinstance(self.y, tuple):
            # lower bound
            if y - self.tol <= self.y[0]:
                exp = SimulationException('Agent is going to fall!')
            # upper bound
            elif y + self.tol >= self.y[1]:
                exp = SimulationException('Agent is going to fall!')

        if self.z + z < self.tol:
            exp = SimulationException('Agent has felt!')

        if exp: raise exp