from world import World
from utils import Supervisor

from deprecation import deprecated

class WebotsWorld(World, Supervisor):
    name = '/krock'

    @deprecated(details='World must be loaded into simulation')
    def __call__(self, *args, **kwargs):
        self.load_world(str(self.path))
