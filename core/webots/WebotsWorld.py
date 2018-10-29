import numpy as np
import time

from world import World
from utils.webots2ros import *
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from tf import transformations

class WebotsWorld(World, Supervisor):
    name = '/krock'

    def __call__(self, *args, **kwargs):
        # TODO check the world name and load if different
        self.load_world(str(self.path))
        # time.sleep(10)

        self.get_world_node()
        self.get_robot_node()

        self.grid = Node.from_def(self.name, 'EL_GRID')
        self.terrain = Node.from_def(self.name, 'TERRAIN')

        self.translation = self.terrain['translation'][0].value

        self.x_dim = self.grid['xDimension'][0].value
        self.x_spac = self.grid['xSpacing'][0].value

        self.y_dim = self.grid['zDimension'][0].value
        self.y_spac = self.grid['zSpacing'][0].value

        self.x = self.x_dim * self.x_spac
        self.y = self.y_dim * self.y_spac
        # define x=(min_x, max_x), y=(min_y,max_y)
        # also add the translation
        self.x = (self.translation.x, self.x + self.translation.x)
        self.y = (self.translation.z, self.y + self.translation.z)

        with open('./webots/children', 'r') as f:
            self.children = f.read()

    def reanimate(self):
        self.get_world_node()
        self.get_robot_node()
        # get the ROBOT node using the NODE API to make our life easier :)
        node = Node.from_def('/krock', 'ROBOT')
        # get the children field that cointas all the joints connections
        h = node['children']
        # remove all children from node ROBOT
        for _ in range(7):
            del node[h]
        # open the original children tree
        # with open('./webots/children', 'r') as f:
        node[h] = self.children
        # restart the simulation and enable the camera
        self.retry_service(self.restart_robot)
        self.retry_service(self.enable_front_camera)
        # update references to GRID and TERRAIN
        self.grid = Node.from_def(self.name, 'EL_GRID')
        self.terrain = Node.from_def(self.name, 'TERRAIN')

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

        random_pose.position.z = h + 0.5
        qto = transformations.quaternion_from_euler(0, 0, 2 * np.pi * np.random.uniform(0, 1), axes='sxyz')
        random_pose.orientation.x = qto[0]
        random_pose.orientation.y = qto[1]
        random_pose.orientation.z = qto[2]
        random_pose.orientation.w = qto[3]
        return random_pose


