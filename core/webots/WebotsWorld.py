from world import World
from utils import Supervisor
import time
from deprecation import deprecated

class WebotsWorld(World, Supervisor):
    name = '/krock'

    def __call__(self, *args, **kwargs):
        self.load_world(str(self.path))
        time.sleep(3)
        self.reset_simulation_physics()