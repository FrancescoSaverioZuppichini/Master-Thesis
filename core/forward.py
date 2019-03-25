import rospy
import pprint
from agent.callbacks import *

from env.spawn.SpawnStragety import FlatGroundSpawnStrategy
from env.webots.krock import Krock, KrockWebotsEnv
import time
import numpy as np

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/bars1.wbt'
# MAP = '/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png'
MAP = '/media/francesco/saetta/test/patches/1550305091.702912.png'
N_STEPS = 4
from utils.webots2ros import Supervisor, Node

rospy.init_node("traversability_simulation")
# create our env

import cv2

image = cv2.imread(MAP)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_REPLICATE)

env = KrockWebotsEnv.from_numpy(
    image,
    '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
    {'height': 10,
     'resolution': 0.02},
    # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
    output_path='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/tmp.wbt')

# env = KrockWebotsEnv.from_image(
#     MAP,
#     '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
#     {'height': 10,
#      'resolution': 0.02},
#     # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
#     output_dir='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/')

# env = KrockWebotsEnv(WORLD_PATH, load_world=True)

# spawn_strategy = FlatGroundSpawnStrategy(MAP, scale = 1 )
# spawn_points = spawn_strategy(k=30, tol=1e-2, size=45)


def spawn_points2webots_pose(spawn_point, env):
    _, orientation = env.random_position
    x,y = spawn_point * 0.02 - 5
    z = env.get_height(x, y)

    pose = [[x, z + 0.5, y], orientation]

    return pose

# env = KrockWebotsEnv(None,
#                      agent_callbacks=[RosBagSaver('/home/francesco/Desktop/querry-high/bags/', topics=['pose'])],
#                      )
# #
# print('Initial observations:')
# pprint.pprint(init_obs)
#
tr = np.array([5,5])
print(np.array(image.shape) / 100 / 2 / 2 )
x,y = -tr + ( np.array(image.shape) * 0.02 / 2) + 0.03
print(x,y)
# print(env.x, env.y)
for i in range(1):
    init_obs = env.reset(pose=[[x , env.get_height(x,y), y],
                               [0,0,0,0]])
    for _ in range(20000):
        obs, r, done, _ = env.step(env.GO_FORWARD)
        pprint.pprint(obs)
        if done: break
    env.agent.die(env)

#     env.step(env.STOP)
#     break
#     # env.reset(pose=pose)
#
