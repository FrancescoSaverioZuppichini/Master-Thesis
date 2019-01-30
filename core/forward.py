import rospy
import pprint

from env.webots.krock import Krock, KrockWebotsEnv

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes1.wbt'
MAP = '/home/francesco/Desktop/querry-big.png'
N_STEPS = 4

rospy.init_node("traversability_simulation")
# create our env
env = KrockWebotsEnv.from_image(
    MAP,
    '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock.wbt',
    {'height': 5,
     'resolution': 0.02},
    output_dir='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/')

# env = KrockWebotsEnv.from_file(WORLD_PATH)
# spawn the robot at a random location
init_obs = env.reset()

print('Initial observations:')
pprint.pprint(init_obs)

print(env.x, env.y)

while True:
    env.reset()
    for _ in range(50):
        obs, r, done, _ = env.step(env.GO_FORWARD)
        pprint.pprint(obs)
        if done: break

    env.step(env.STOP)
