#!/usr/bin/env python

import sys
import rospy
import os
import time
import pickle
import numpy as np
import math

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Header

from tf import transformations

class SingleRecord:
    def __init__(self):
        self.rate = rospy.Rate(10)

        # constants of the robot
        self.model_name = "krock"

        # constants depending on the map (we assume we have a square map with origin at the center, ergo size would be 2x times 2y.
        # These are default values:
        self.map_max_x = 5.0
        self.map_max_y = 5.0
        self.map_max_z = 1.0

        self.pub_status = rospy.Publisher("/krock/status", String, queue_size=1)
        self.pub_spawn_pose = rospy.Publisher("/krock/spawn_pose", PoseStamped)

        rospy.Subscriber("/krock/pose", PoseStamped, self.callback_pose)

    def generate_random_pose(self):
        # x,y (z will be fixed as the max_hm_z so that the robot will drop down), gamma as orientation
        rn = np.random.random_sample((3,))
        random_pose = Pose()
        random_pose.position.x = 2 * self.map_max_x *rn[0] - self.map_max_x
        random_pose.position.y = 2 * self.map_max_x *rn[1] - self.map_max_x
        random_pose.position.z = self.map_max_z * 1.0 # spawn on the air
        qto = transformations.quaternion_from_euler(0, 0, 2*math.pi * rn[1], axes='sxyz')
        random_pose.orientation.x = qto[0]
        random_pose.orientation.y = qto[1]
        random_pose.orientation.z = qto[2]
        random_pose.orientation.w = qto[3]
        return random_pose

    def generate_random_stamped_pose(self):
        pose = self.generate_random_pose()
        pose_stamped = PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose
        return pose_stamped

    def publish_spawn_pose(self):
        rospy.loginfo("Publishing spawing pose.")
        self.pub_spawn_pose.publish(self.initial_pose)

    def setup_new_simulation(self, world_name):
        self.world_name = world_name
        self.initial_pose = self.generate_random_stamped_pose()
        rospy.loginfo("Random pose generated: %s \n(%s) \n(%s)", self.initial_pose.header, self.initial_pose.pose.position, self.initial_pose.pose.orientation)
        self.publish_spawn_pose()


    def callback_pose(self, data):
        rospy.loginfo("Pos data received")
        return 0

    def run_simulation(self, world_name, number_tries, dataset_name, duration=10):
        self.setup_new_simulation(world_name)


# if __name__ == "__main__":
#     if len(sys.argv) == 4:
#         world_name = sys.argv[1]
#         number_tries = sys.argv[2]
#         dataset_name = sys.argv[3]
#     else:
#         rospy.logerr("record_single_trajectory world_name number_tries dataset_name")
#         sys.exit(1)

rospy.init_node("record_single_trajectory")
r =rospy.Rate(10)
sim = SingleRecord()
sim.generate_random_pose()
sim.run_simulation(0, 1, 'test')
r.sleep()
sim.run_simulation(0, 1, 'test')
r.sleep()

