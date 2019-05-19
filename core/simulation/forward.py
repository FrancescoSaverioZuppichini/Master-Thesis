from simulation.agent.callbacks import *
from simulation.env.webots.krock import KrockWebotsEnv
from tf import transformations
from simulation.env.spawn import FlatGroundSpawnStrategy, spawn_points2webots_pose

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/bars1.wbt'
MAP = '/home/francesco/Documents/Master-Thesis/core/maps/test/flat.png'
# MAP = '/media/francesco/saetta/krock-dataset/test_with_obstacles/wall.png'

# MAP = '/home/francesco/Desktop/center.png'
N_STEPS = 4

rospy.init_node("traversability_simulation")
# create our env
import random

# image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_REPLICATE)

# env = KrockWebotsEnv.from_numpy(
#     image,
#     '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
#     {'height': 10,
#      'resolution': 0.02},
#     # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
#     output_path='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/tmp.wbt')
#
# env = KrockWebotsEnv.from_image(
#     MAP,
#     '/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock.wbt',
#     {'height': 1,
#      'resolution': 0.02 },
#     # agent_callbacks=[RosBagSaver('/media/francesco/saetta/krock-dataset/test_with_obstacle_in_center/bags',
#     #                              topics=['pose'])],
#     output_dir='/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock2_ros/worlds/')
# #
# env = KrockWebotsEnv(WORLD_PATH, load_world=True)

# spawn_strategy = FlatGroundSpawnStrategy(MAP, scale = 1, debug=True)
# spawn_points = spawn_strategy(k=20, tol=1e-2, size=45)


def spawn_points2webots_pose(spawn_point, env):
    _, orientation = env.random_position
    x,y = spawn_point
    # x,y = 250, 450
    z = env.get_height(x, y)
    print(z)
    pose = [[(x * 0.02) - 5, z + 0.4, (y * 0.02) - 5], orientation]

    return pose
#
env = KrockWebotsEnv(None,
                     agent_callbacks=[RosBagSaver('/media/francesco/saetta/krock-dataset/tr/bags',
                                                  topics=['pose'])],
                     )

# env.reset(spawn=False)
# import rospy
#
# z = env.get_height(250, 500)
# print(z)
# # print(env.x, env.y, h)
# for i in range(10):
#     # spawn_point = random.choice(spawn_points)
#     spawn_point = [random.randint(22, 513 - 22), random.randint(22, 513 - 22)]
#     # spawn_points2webots_pose(spawn_point, env)
#     init_obs = env.reset(pose=spawn_points2webots_pose(spawn_point, env))
#     time.sleep(0.2)
# #
# #

file_name = 'test'

env.agent()
env.reset(spawn=False)

meta = pd.DataFrame(data={'filename': [file_name],
                          'map': ['bumps3-rocks1'],
                          'height': [1]})

meta.to_csv('/media/francesco/saetta/krock-dataset/tr/meta.csv')
elapsed = 0
start = time.time()
# #
while elapsed <= 10:
# #     # print(time.time())
    elapsed = time.time() - start
#
    obs, r, done, _ = env.step(env.GO_FORWARD)
    # pass
#
# #
env.step(env.STOP)
# #
env.agent.die(env, file_name)
