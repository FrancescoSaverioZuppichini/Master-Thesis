import numpy as np
import cv2
import os
import gym

from utilities.webots2ros import Supervisor, Node
from geometry_msgs.msg import Pose
from tf import transformations
from .utils import image2webots_terrain
from matplotlib.pyplot import imshow

from .utils import image2webots_terrain
from os import path


class WebotsEnv(gym.Env, Supervisor):

    def __init__(self, world_path, load_world=True, children_path=None):
        self.world_path, self.children_path = world_path, children_path
        # TODO refactor
        if load_world: self.load_world(self.world_path)

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

        self.children=None

        if children_path is not None:
            with open(self.children_path, 'r') as f:
                self.children = f.read()


    def reanimate(self):
        if self.children is None: raise Exception('No children specified!')
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

    def get_height(self, x, y):
        idx = int(x + (self.x_dim * y))
        h = self.grid['height'][idx].value

        return h

    @property
    def random_position(self):
        rx = np.random.uniform(*self.x)
        ry = np.random.uniform(*self.y)

        random_pose = Pose()
        random_pose.position.x = rx
        random_pose.position.y = ry
        h = self.get_height(rx, ry)

        random_pose.position.z = h + 0.5
        qto = transformations.quaternion_from_euler(0, 0, 2 * np.pi * np.random.uniform(0, 1), axes='sxyz')

        position = [rx, h + 0.5, ry]

        return position, [qto[0], qto[2], qto[1], qto[3]]

    def spawn(self, agent, pose=None, *args, **kwargs):
        pose = self.random_position if pose is None else pose

        position, orientation = pose

        self.set_robot_position(*position)

        self.set_robot_orientation(*orientation)

        self.reset_node_physics(self.robot_node)

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        return cls(file_path, *args, **kwargs)

    @classmethod
    def from_image(cls, image_path, src_world, config, output_dir, *args, **kwargs):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        filename_w_ext = os.path.basename(image_path)
        filename, file_extension = os.path.splitext(filename_w_ext)

        output_path = os.path.normpath(output_dir + '/' + filename + '.wbt')

        return cls.from_numpy(image, src_world, config, output_path, *args, **kwargs)

    @classmethod
    def from_numpy(cls, image_np, src_world, config, output_path, *args, **kwargs):
        world_path = image2webots_terrain(image_np, src_world, config, output_path)

        return cls(world_path, *args, **kwargs)
