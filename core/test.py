import rospy
import numpy as np

from tf import transformations

from agent.krock import Krock
from agent.callbacks import RosBagSaver

rospy.init_node("record_single_trajectory")

map_max_x = 5.0
map_max_y = 5.0
map_max_z = 1.0
# TODO this stuff should go in utils or in simlation class
def generate_random_pose():
    # x,y (z will be fixed as the max_hm_z so that the robot will drop down), gamma as orientation
    rn = np.random.random_sample((3,))
    random_pose = Pose()
    random_pose.position.x = 2 * map_max_x *rn[0] - map_max_x
    random_pose.position.y = 2 * map_max_x *rn[1] - map_max_x
    random_pose.position.z = map_max_z * 1.0 # spawn on the air
    qto = transformations.quaternion_from_euler(0, 0, 2*np.pi * rn[1], axes='sxyz')
    random_pose.orientation.x = qto[0]
    random_pose.orientation.y = qto[1]
    random_pose.orientation.z = qto[2]
    random_pose.orientation.w = qto[3]
    return random_pose

r = rospy.Rate(hz=0.5)
k = Krock()

k.add_callback(RosBagSaver('./data.bag'))
k()
# r.sleep()
# k.spawn()

while not rospy.is_shutdown():
    k.spawn()
#     # k.move()
    r.sleep()

rospy.on_shutdown(k.on_shut_down)
