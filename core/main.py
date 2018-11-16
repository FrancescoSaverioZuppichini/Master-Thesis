from agent.callbacks import *
from webots.krock import Krock

from simulation.conditions import *

from webots.krock.KrockWebotsEnv import KrockWebotsEnv
from webots import *

from parser import args

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

if args.maps == None:  args.maps = [WORLD]

agent = None

# TODO main file is polluted. Create something like Topology .from_args that returns agent, worlds and sim

def make_env(map):
    env = None
    if args.engine == 'webots':
        if args.robot == 'krock':
            src_world = path.abspath('./webots/krock/krock.wbt')

            env = KrockWebotsEnv.from_image(
                map,
                path.abspath('./webots/krock/krock.wbt'),
                {'height': 1,
                 'resolution': 0.02},
                output_dir=path.abspath('./webots/krock/krock2_ros/worlds/'),
                agent_callbacks=[RosBagSaver(args.save_dir, topics=['pose'])]
            )

    return env

N_SIM = 1000

b = range(N_SIM)

start = time.time()
print('')
print(len(args.maps))

for map in args.maps:
    env = make_env(map)

    for i in range(N_SIM):
        # TODO as always the reanimation breaks something
        # if i % 20 == 0:
        #     rospy.loginfo('Reanimate robot')
        #     w.reanimate()
        env.reset()

        for i in range(200):
            env.render()
            obs, r, done, _ = env.step(env.GO_FORWARD)
            if done: break
        print('Done after {}'.format(i))
        # we want to store at each spawn
        env.agent.die(env)


    end = time.time() - start

    rospy.loginfo('Iter={:} Elapsed={:.2f}'.format(str(i), end))
# except Exception as e:
#     print(e)
#     # rospy.signal_shutdown('KeyboardInterrupt')
