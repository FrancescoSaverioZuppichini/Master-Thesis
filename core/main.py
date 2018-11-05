import rospy
import time

from agent.callbacks import *
from webots.krock import Krock

from simulation import BasicSimulation
from simulation.callbacks import *

from webots import *

from parser import args

N_SIM = args.n_sim
SIM_TIME = args.time
WORLD = args.world

rospy.init_node("traversability_simulation")

# TODO the robot should be selected based on the parser
w = WebotsWorld.from_image(
    '/home/francesco/Documents/Master Thesis/core/webots/map_generation/bumps2.png',
    '/home/francesco/Documents/krock-sim/krock/krock2_ros/worlds/krock2_camera.wbt',
    { 'height' : 2,
      'resolution' : 0.02},
    output_path='/home/francesco/Documents/krock-sim/krock/krock2_ros/worlds/test.wbt')

# w = WebotsWorld.from_file(WORLD)
w()

# TODO the robot should be selected based on the parser
def create_agent():
    krock = Krock()
    krock.add_callback(RosBagSaver('./data',
                                   topics=['pose']))

    # krock.add_callback(RosBagSaver('./data/{}.bag'.format(time.time()),
    #                                topics=['pose']))
    krock()

    return krock

sim = BasicSimulation(name=args.robot)
sim.add_callbacks([Alarm(stop_after_s=SIM_TIME),
                   OutOfMap(x=w.x, y=w.y)
                   ])


b = range(N_SIM)

start = time.time()
print('')

for iter, _ in enumerate(b):
    if iter % 5 == 0: w.reanimate()
    a = create_agent()

    sim(world=w,
        agent=a)
    end = time.time() - start

    rospy.loginfo('Iter={:} Error={:} Elapsed={:.2f}'.format(str(iter), sim.history['error', -1], end))
