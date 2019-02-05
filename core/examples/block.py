from agent.callbacks import *
from env.webots.krock.KrockWebotsEnv import KrockWebotsEnv

rospy.init_node("traversability_simulation")

start = time.time()
map = 'flat'
map_name, _ = path.splitext(path.basename(map))
bag_save_path = path.normpath('.' + '/' + map_name +'-block')

env = KrockWebotsEnv('/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/flat.wbt',
                     agent_callbacks=[RosBagSaver(bag_save_path, topics=['pose'])])

while True:
    env.reset()
    for _ in range(50):
        obs, r, done, _ = env.step(env.STOP)
        if done: break
    env.agent.die(env)

