from simulation.agent.callbacks import *

from simulation.env.webots.krock import KrockWebotsEnv
from tf import transformations

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/bars1.wbt'
MAP = '/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png'
# MAP = '/home/francesco/Desktop/center.png'
N_STEPS = 4

rospy.init_node("traversability_simulation")
# create our env


# image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_REPLICATE)

# env = KrockWebotsEnv.from_numpy(
#     image,
#     '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock_no_tail.wbt',
#     {'height': 10,
#      'resolution': 0.02},
#     # agent_callbacks=[RosBagSaver('~/Desktop/querry-high/bags', topics=['pose'])],
#     output_path='/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/tmp.wbt')

# env = KrockWebotsEnv.from_image(
#     MAP,
#     '/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock_no_tail.wbt',
#     {'height': 10,
#      'resolution': 0.02 },
#     agent_callbacks=[RosBagSaver('/media/francesco/saetta/quarry-ramp/bags',
#                                  topics=['pose'])],
#     output_dir='/home/francesco/Documents/Master-Thesis/core/simulation/env/webots/krock/krock2_ros/worlds/')

# env = KrockWebotsEnv(WORLD_PATH, load_world=True)

# spawn_strategy = FlatGroundSpawnStrategy(MAP, scale = 1 )
# spawn_points = spawn_strategy(k=30, tol=1e-2, size=45)


def spawn_points2webots_pose(spawn_point, env):
    _, orientation = env.random_position
    x,y = spawn_point * 0.02 - 5
    z = env.get_height(x, y)

    pose = [[x, z + 0.5, y], orientation]

    return pose

env = KrockWebotsEnv(None,
                     agent_callbacks=[RosBagSaver('/media/francesco/saetta/quarry-ramp/bags',
                                                  topics=['pose'])],
                     )

env.agent()
env.reset(spawn=False)
# print(env.x, env.y, h)
for i in range(1):
    # spawn_point = random.choice(spawn_points)

    # init_obs = env.reset(pose=spawn_points2webots_pose(spawn_point, env))

    # init_obs = env.reset(pose=[[x , h + 0.2, y],
    #                            qto])

    for _ in range(200):
        # time.sleep(0.01)
        obs, r, done, _ = env.step(env.GO_FORWARD)
        # pprint.pprint(obs)
        # if done: break
    env.agent.die(env)

#     env.step(env.STOP)
#     break
#     # env.reset(pose=pose)
#
