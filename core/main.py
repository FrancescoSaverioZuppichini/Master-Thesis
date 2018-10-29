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

# rospy.on_shutdown(lambda : print('Bye'))

w = WebotsWorld(file_path=WORLD)
w()

def create_agent():
    krock = Krock()
    krock.add_callback(RosBagSaver('./data',
                                   topics=['pose']))
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
