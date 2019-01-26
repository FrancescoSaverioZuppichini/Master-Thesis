import rospy
import pprint

from env.webots.krock import Krock, KrockWebotsEnv

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/flat.wbt'
N_STEPS = 4

rospy.init_node("traversability_simulation")
# create our env
env = KrockWebotsEnv.from_file(WORLD_PATH)
# spawn the robot at a random location
init_obs = env.reset()

print('Initial observations:')
pprint.pprint(init_obs)

while True:
    env.reset()
    while True:
        obs, r, done, _ = env.step(env.GO_FORWARD)
        if done: break

    env.step(env.STOP)