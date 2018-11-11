import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException
# TODO an extractor function should be added to allow more flexibility into the agent state
class OutOfMap(SimulationCallback):
    def __init__(self, tol=-.01):
        self.tol = tol

    @staticmethod
    def is_inside(world, agent, tol=-.01):
        pose = agent.state['pose']
        pos = pose.pose.position

        x, y, z = pos.x, pos.y, pos.z

        def check_if_inside(to_check, bounds):
            lower, upper = bounds
            # TODO bad raising in if statement
            if to_check - tol <= lower:
                return False
            # upper bound
            elif to_check + tol >= upper:
                return False

            return True

        has_not_fall = world.z + z >= tol

        is_inside_x = check_if_inside(x, world.x)
        is_inside_y = check_if_inside(y, world.y)

        return  is_inside_x and is_inside_y and has_not_fall

    def tick(self, sim, world, agent, *args, **kwargs):
        if 'pose' not in agent.state: return

        if not self.is_inside(world, agent, self.tol): raise SimulationException('Fall.')
