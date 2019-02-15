import rospy
import pprint
from agent.callbacks import *

from env.spawn.SpawnStragety import FlatGroundSpawnStrategy
from env.webots.krock import Krock, KrockWebotsEnv
import time
import numpy as np

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/bars1.wbt'
MAP = './maps/train/steps3.png'
N_STEPS = 4
from utils.webots2ros import Supervisor, Node

rospy.init_node("traversability_simulation")
# create our env
env = KrockWebotsEnv.from_image(
    MAP,
    '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
    {'height': 1,
     'resolution': 0.02},
    agent_callbacks=[RosBagSaver('~/Desktop/krock_upside_down/', topics=['pose'])],
    output_dir='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/')

# env = KrockWebotsEnv(WORLD_PATH, load_world=True, agent_callbacks=[RosBagSaver('~/Desktop/test_tail/', topics=['pose'])])

spawn_strategy = FlatGroundSpawnStrategy('./maps/train/bars1.png')

spawn_points = spawn_strategy(k=30, tol=1e-2, size=45)


def spawn_points2webots_pose(spawn_point, env):
    _, orientation = env.random_position
    x,y = spawn_point * 0.02 - 5
    z = env.get_height(x, y)

    pose = [[x, z + 0.5, y], orientation]

    return pose

# env = KrockWebotsEnv(None,
#                      # agent_callbacks=[RosBagSaver('/home/francesco/Desktop/krock_upside_down', topics=['pose'])]
#                      )
# #
# print('Initial observations:')
# pprint.pprint(init_obs)
#
# print(env.x, env.y)
# env.step(env.STOP)
for i in range(100):
    init_obs = env.reset(pose=spawn_points2webots_pose( spawn_points[i], env))
    for _ in range(100):
        obs, r, done, _ = env.step(env.GO_FORWARD)
        pprint.pprint(obs)
        if done: break
    # env.agent.die(env)

#     env.step(env.STOP)
#     break
#     # env.reset(pose=pose)
#
