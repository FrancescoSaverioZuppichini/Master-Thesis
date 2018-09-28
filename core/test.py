import rospy
from agent.krock import Krock
from agent.callbacks import RosBagSaver

rospy.init_node("record_single_trajectory")

r = rospy.Rate(hz=10)

k = Krock()

k.add_callback(RosBagSaver('./data.bag'))
k()

while not rospy.is_shutdown():
    k.move(2)
    r.sleep()

rospy.on_shutdown(k.on_shut_down)
