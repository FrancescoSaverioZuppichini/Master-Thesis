import rospy
import pprint
from agent.callbacks import *

from env.webots.krock import Krock, KrockWebotsEnv
import time

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes1.wbt'
MAP = './maps/train/slope_rocks1.png'
N_STEPS = 4
from utils.webots2ros import Supervisor, Node

rospy.init_node("traversability_simulation")
# create our env
# env = KrockWebotsEnv.from_image(
#     MAP,
#     '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
#     {'height': 5,
#      'resolution': 0.02},
#     agent_callbacks=[RosBagSaver('~/Desktop/krock_upside_down/', topics=['pose'])],
#     output_dir='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/')

# env = KrockWebotsEnv(WORLD_PATH, load_world=True, agent_callbacks=[RosBagSaver('~/Desktop/test_tail/', topics=['pose'])])
# spawn the robot at a random location

#
# # get the ROBOT node using the NODE API to make our life easier :)
# node = Node.from_def('/krock', 'ROBOT')
# # get the children field that cointas all the joints connections
# h = node['children']
#
# for i in range(7):
#     print(h[i])
# pose = [[0,-3.6,-3.5],[0, 1, 0, -1.57]]

env = KrockWebotsEnv(None, agent_callbacks=[RosBagSaver('/home/francesco/Desktop/krock_upside_down', topics=['pose'])])

init_obs = env.reset(spawn=False)
#
print('Initial observations:')
pprint.pprint(init_obs)
#
# print(env.x, env.y)
env.step(env.STOP)

for _ in range(100):
    obs, r, done, _ = env.step(env.STOP)
    pprint.pprint(obs)
    if done: break
env.agent.die(env)

#     env.step(env.STOP)
#     break
#     # env.reset(pose=pose)
#
