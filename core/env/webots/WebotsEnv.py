import numpy as np
import cv2
import os

import gym

from utils.webots2ros import Supervisor, Node
from geometry_msgs.msg import Pose
from tf import transformations
from .utils import image2webots_terrain
from matplotlib.pyplot import imshow

from .utils import image2webots_terrain
from os import path


class WebotsEnv(gym.Env, Supervisor):
    name = '/krock'

    def __init__(self, world_path, load_world=True,*args, **kwargs):
        self.world_path = world_path
        # TODO refactor
        print(self.world_path)
        # self.load_world('/home/francesco/Documents/Master-Thesis/core/webots/' + self.world_path)
        if load_world: self.load_world( self.world_path)

        self.retry_service(self.get_world_node)
        self.retry_service(self.get_robot_node)
        self.retry_service(self.enable_front_camera)

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
        self.z = 0

        with open(path.abspath('./env/webots/children'), 'r') as f:
            self.children = f.read()

    def reanimate(self):
        # get the ROBOT node using the NODE API to make our life easier :)
        node = Node.from_def('/krock', 'ROBOT')
        # get the children field that cointas all the joints connections
        h = node['children']
        # remove all children from node ROBOT
        for _ in range(7):
            del node[h]
        node[h] = self.children
        # restart the simulation and enable the camera
        self.retry_service(self.get_world_node)
        self.retry_service(self.restart_robot)
        self.retry_service(self.get_robot_node)
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
        idx = int(
            ((rx + abs(self.translation.x)) // self.x_spac) + (1600 * ((ry + abs(self.translation.z)) // self.y_spac)))

        h = self.grid['height'][idx].value

        random_pose.position.z = h + 0.5
        qto = transformations.quaternion_from_euler(0, 0, 2 * np.pi * np.random.uniform(0, 1), axes='sxyz')
        random_pose.orientation.x = qto[0]
        random_pose.orientation.y = qto[1]
        random_pose.orientation.z = qto[2]
        random_pose.orientation.w = qto[3]
        return random_pose

    def spawn(self, agent, position=NOne, *args, **kwargs):
        if type(pos) is list: pos = Pose

        pos = self.random_position if pos == None else pos

        self.set_robot_position(x=pos.position.x,
                                y=pos.position.z,
                                z=pos.position.y)

        self.set_robot_orientation(x=pos.orientation.x,
                                   y=pos.orientation.z,
                                   z=pos.orientation.y,
                                   w=pos.orientation.w)

        self.reset_node_physics(self.robot_node)

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        return cls(file_path, *args, **kwargs)

    @classmethod
    def from_image(cls, image_path, src_world, config, output_dir, *args, **kwargs):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        filename_w_ext = os.path.basename(image_path)
        filename, file_extension = os.path.splitext(filename_w_ext)

        output_path = os.path.normpath(output_dir + '/' + filename + '.wbt')

        return cls.from_numpy(image, src_world, config, output_path, *args, **kwargs)

    @classmethod
    def from_numpy(cls, image_np, src_world, config, output_path, *args, **kwargs):
        world_path = image2webots_terrain(image_np, src_world, config, output_path)

        return cls(world_path, *args, **kwargs)

    @classmethod
    def from_dir(cls, dir):
        # TODO
        files = os.listdir(dir)
