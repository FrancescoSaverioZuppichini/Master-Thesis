import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException
# TODO this is a naive implementation, a more robust approach
# should be tracing the z and check how much it changed in time
class FallDetector(SimulationCallback):
    def __init__(self, ground_z=0, tol=-.1**2):
        self.ground_z = ground_z
        self.tol = tol

    def tick(self, sim, world, agent, *args, **kwargs):
        pose = agent.state['pose']
        z = pose.pose.position.z
        has_felt = self.ground_z + z < self.tol
        if has_felt: raise SimulationException('Agent has felt!')