import numpy as np
import time

from world import World
from utils.webots2ros import *
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from tf import transformations

class WebotsWorld(World, Supervisor):
    name = '/krock'

    def __call__(self, *args, **kwargs):
        # self.load_world(str(self.path))
        # time.sleep(10)
        self.reset_simulation_physics()
        # self.restart_robot()
        self.grid = Node.from_def(self.name, 'EL_GRID')
        self.terrain = Node.from_def(self.name, 'TERRAIN')

        self.translation =self.terrain['translation'][0].value

        self.x_dim = self.grid['xDimension'][0].value

        self.x_spac = self.grid['xSpacing'][0].value

        self.y_dim = self.grid['zDimension'][0].value
        self.y_spac = self.grid['zSpacing'][0].value

        self.x = self.x_dim * self.x_spac
        self.y = self.y_dim * self.y_spac

        self.x = (self.translation.x, self.x + self.translation.x)
        self.y = (self.translation.z, self.y + self.translation.z)

    @property
    def random_position(self):
        rx = np.random.uniform(*self.x)
        ry = np.random.uniform(*self.y)


        random_pose = Pose()
        random_pose.position.x = rx
        random_pose.position.y = ry
        # to get the 2d index in 1d matrix x + width * y
        idx = int(((rx + abs(self.translation.x)) // self.x_spac )+ (1600 * ((ry + abs(self.translation.z))// self.y_spac)))

        h = self.grid['height'][idx].value

        random_pose.position.z = h + 1
        print(h)
        # print(random_pose.position.z )
        qto = transformations.quaternion_from_euler(0, 0, 2 * np.pi * np.random.uniform(0, 1), axes='sxyz')
        random_pose.orientation.x = qto[0]
        random_pose.orientation.y = qto[1]
        random_pose.orientation.z = qto[2]
        random_pose.orientation.w = qto[3]
        return random_pose


