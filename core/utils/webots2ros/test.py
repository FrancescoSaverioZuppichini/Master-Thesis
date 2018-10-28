import sys
import rospy
import time
import numpy as np
import math
from geometry_msgs.msg import Quaternion, Vector3
from webots_ros.srv import *
from agent.krock import Krock
from utils.webots2ros.Supervisor import *

rospy.init_node("traversability_simulation")

s = Supervisor()
s.name = '/krock'
s.get_world_node()
s.get_robot_node()

node = Node.from_def('/krock', 'ROBOT')

h = node['children']

for _ in range(7):
    del node[h]

with open('../../webots/children', 'r') as f:
    node[h] = f.read()


print(s.retry_service(s.restart_robot))
print(s.retry_service(s.enable_front_camera))

grid = Node.from_def(s.name, 'EL_GRID')

print(grid['height'][2].value)


node = Node.from_def('/krock', 'ROBOT')

h = node['children']
#
# s = Supervisor()
# s.name = '/krock'
s.get_world_node()
s.get_robot_node()


for _ in range(7):
    del node[h]

with open('../../webots/children', 'r') as f:
    node[h] = f.read()


print(s.retry_service(s.restart_robot))
print(s.retry_service(s.enable_front_camera))

# s.restart_robot()
grid = Node.from_def(s.name, 'EL_GRID')
print(grid['height'][2].value)

k = Krock()
k()
for _ in range(3):
    k.act(None)
    time.sleep(1)

k.stop()


# rospy.init_node("traversability_simulation")

# for _ in range(7):
#     #
#     del node[h]
# with open('../../webots/children', 'r') as f:
#     node[h] = f.read()

#
# s.get_world_node()
# s.get_robot_node()
# s.restart_robot()
#
# print(s.retry_service(s.enable_front_camera))
#
# grid = Node.from_def(s.name, 'EL_GRID')
# print(grid['height'][2].value)

# s.reset_simulation()
# s.restart_robot()

    # while True:
    #     try:
    #         print(s.enable_front_camera())
    #     except rospy.service.ServiceException:
    #         time.sleep(0.1)
    # print('supervisor')
    # s = Supervisor()
    # s.name = '/krock'
    # s.get_world_node()
    # s.get_robot_node()

    # s.enable_front_camera()
# node[h] =    'DEF GPS_FGIRDLE GPS { \
#       name "gps_fgirdle" \
#     } \
#     DEF IMU InertialUnit { \
#       name "IMU" \
#     }'

# del node[h]