import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException
# TODO an extractor function should be added to allow more flexibility into the agent state
class OutOfMap(SimulationCallback):
    def __init__(self, tol=-.01):
        self.tol = tol

    def tick(self, sim, world, agent, *args, **kwargs):
        if 'pose' not in agent.state: return

        pose = agent.state['pose']
        pos = pose.pose.position
        ori = pose.pose.orientation
        #
        # if ori.w < 0:
        #     raise SimulationException('Upside down.')

        self.x, self.y, self.z = world.x, world.y, world.z

        x, y, z = pos.x, pos.y, pos.z

        def check_if_inside(to_check, bounds):
            lower, upper = bounds
            # TODO bad raising in if statement
            if to_check - self.tol <= lower:
                raise SimulationException('Fall.')
            # upper bound
            elif to_check + self.tol >= upper:
                raise SimulationException('Fall.')

        check_if_inside(x, self.x)
        check_if_inside(y, self.y)

        if self.z + z < self.tol: raise SimulationException('Fall.')
