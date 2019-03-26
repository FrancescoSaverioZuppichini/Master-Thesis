import rospy
import pprint
from agent.callbacks import *

from env.spawn.SpawnStragety import FlatGroundSpawnStrategy
from env.webots.krock import Krock, KrockWebotsEnv
import time
import numpy as np
from tf import transformations

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/bars1.wbt'
# MAP = '/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png'
MAP = '/home/francesco/Desktop/center.png'
N_STEPS = 4
from utils.webots2ros import Supervisor, Node

rospy.init_node("traversability_simulation")
# create our env
from geometry_msgs.msg import PoseStamped

import cv2

image = cv2.imread(MAP)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


from estimators.patches import *

p = BarPatch((125,125))
p()

image = p.hm

# image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_REPLICATE)

# env = KrockWebotsEnv.from_numpy(
#     image,
#     '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
#     {'height': 10,
#      'resolution': 0.02},
#     # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
#     output_path='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/tmp.wbt')

env = KrockWebotsEnv.from_image(
    MAP,
    '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock.wbt',
    {'height': 1,
     'resolution': 0.02 },
    agent_callbacks=[RosBagSaver('/home/francesco/Desktop/krock-center-tail/bags/center/',
                                 topics=['pose'])],
    output_dir='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/')

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
print(np.array(100) / 100 / 2 / 2 )
x,y = -tr + ( np.array(100) * 0.02 / 2)
print(x,y)

qto = transformations.quaternion_from_euler(0, 0, 0, axes='sxyz')

qto = [qto[0], qto[2], qto[1], qto[3]]

h = env.get_height(x, y)

env.agent()
print(env.x, env.y, h)
for i in range(1):
    init_obs = env.reset(pose=[[x , h + 0.2, y],
                               qto])
    for _ in range(50):
        env.agent.sleep()
        # time.sleep(0.01)
        # obs, r, done, _ = env.step(env.STOP)
        # pprint.pprint(obs)
        # if done: break
    env.agent.die(env)

#     env.step(env.STOP)
#     break
#     # env.reset(pose=pose)
#
