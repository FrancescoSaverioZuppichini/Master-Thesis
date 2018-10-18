from world import World
from utils.webots2ros import *
import time

class WebotsWorld(World, Supervisor):
    name = '/krock'

    def __call__(self, *args, **kwargs):
        self.load_world(str(self.path))
        time.sleep(5)

        self.grid = Node.from_def(self.name, 'EL_GRID')

        self.x_dim = self.grid['xDimension'][0].value
        self.x_spac = self.grid['xSpacing'][0].value

        self.y_dim = self.grid['zDimension'][0].value
        self.y_spac = self.grid['zSpacing'][0].value
