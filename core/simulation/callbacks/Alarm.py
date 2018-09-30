import threading
import time

from .SimulationCallback import SimulationCallback

class Alarm(SimulationCallback):
    def __init__(self, stop_after_s=60):
        self.stop_after_s = stop_after_s

    def on_start(self, sim, *args, **kwargs):
        self.start = time.time()

    def tick(self, sim, *args, **kwargs):
        elapsed = (time.time() - self.start)
        sim.should_stop = elapsed >= self.stop_after_s