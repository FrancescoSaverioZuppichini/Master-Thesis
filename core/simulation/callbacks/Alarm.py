import threading
import time

from .SimulationCallback import SimulationCallback
from ..errors import SimulationException

class Alarm(SimulationCallback):
    def __init__(self, stop_after_s=60):
        self.stop_after_s = stop_after_s

    def on_start(self, sim, *args, **kwargs):
        self.start = time.time()

    def tick(self, sim, *args, **kwargs):
        elapsed = (time.time() - self.start)
        should_stop = elapsed >= self.stop_after_s
        if should_stop: raise SimulationException('Time elapsed!')