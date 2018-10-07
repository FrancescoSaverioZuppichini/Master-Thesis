import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException
# TODO this is a naive implementation, a more robust approach
# should be tracing the z and check how much it changed in time
class OutOfMap(SimulationCallback):
    def __init__(self, x, y, z=0, tol=-.01):
        self.x, self.y, self.z = x, y, z
        self.tol = tol

    def tick(self, sim, world, agent, *args, **kwargs):
        pose = agent.state['pose']
        pos = pose.pose.position
        x, y, z = pos.x, pos.y, pos.z

        def check_if_inside(to_check, bounds):
            lower, upper = bounds
            if to_check - self.tol <= lower:
                raise SimulationException('Agent is going to fall!')
            # upper bound
            elif to_check + self.tol >= upper:
                raise SimulationException('Agent is going to fall!')

        check_if_inside(x, self.x)
        check_if_inside(y, self.y)

        if self.z + z < self.tol: raise SimulationException('Agent has felt!')
