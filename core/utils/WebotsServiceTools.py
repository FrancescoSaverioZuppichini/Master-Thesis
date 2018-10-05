#!/usr/bin/env python
#
# implementation of service calls using the supervisor API of webots
#
import sys
import rospy
import time
import numpy as np
import math

# API reference for supervisor services exposed from webots
# https://www.cyberbotics.com/doc/reference/supervisor

# These message type definitions are needed to request or set values
# to some webots services. These requirements are found in the API
# reference.
from geometry_msgs.msg import Quaternion, Vector3
from utils.ros import ros_service
# These service are need for calling and receiving services
# If a new service type is needed, verify the API reference and add it
# here. I have included such files in the srv folder of this package
# If the srv is not found, verify that it appears in the service
# generation segment in the CMakeLists.txt of this package
from webots_ros.srv import get_bool, get_int, get_uint64, set_int, node_get_field, field_get_node, field_get_vec3f, node_get_position, node_get_orientation, field_set_vec3f, field_set_rotation

class WebotsServiceTools:
    def __init__(self, model_name, verbose=False):
        # constants of the robot
        self.model_name = model_name
        self.verbose = verbose

    def get_world_node(self):
        service = self.model_name + '/supervisor/get_root'
        world = None
        try:
            get_world = rospy.ServiceProxy(service, get_uint64)
            world = get_world()
        except rospy.ServiceException as e:
            print ("Service call failed: ", e)
        finally:
            return world

    def reload_world(self):
        service = self.model_name + '/supervisor/world_reload'
        rospy.wait_for_service(service)
        try:
            res = rospy.ServiceProxy(service, get_uint64)
            print(res)
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def reset_rimulation(self):
        service = self.model_name + '/supervisor/simulation_reset'
        rospy.wait_for_service(service)
        try:
            res = rospy.ServiceProxy(service, get_uint64)
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def get_robot_node(self):
        service = self.model_name+'/supervisor/get_self'
        if self.verbose: rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            get_self = rospy.ServiceProxy(service, get_uint64)
            self.robot_node = get_self()
            print (self.robot_node)
        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def get_robot_position(self):
        service = self.model_name+'/supervisor/node/get_position'
        if self.verbose: rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_position = rospy.ServiceProxy(service, node_get_position)
            position = request_position(self.robot_node.value)
            print (position)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def get_robot_orientation(self):
        service = self.model_name+'/supervisor/node/get_orientation'
        if self.verbose: rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_position = rospy.ServiceProxy(service, node_get_orientation)
            orientation = request_position(self.robot_node.value)
            print (orientation)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def set_robot_position(self, x, y, z):
        service = self.model_name+'/supervisor/node/get_field'
        if self.verbose: rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_field = rospy.ServiceProxy(service, node_get_field)
            field = request_field(self.robot_node.value, 'translation')
            print (field)

            other_service = self.model_name+'/supervisor/field/set_vec3f'
            if self.verbose: rospy.loginfo("Waiting for service %s", service)
            rospy.wait_for_service(other_service)
            try:
                set_field = rospy.ServiceProxy(other_service, field_set_vec3f)
                new_position = Vector3()
                # webots position, also called translation is different from ROS
                # x -> forward, z-> right, y -> upward
                new_position.x = x
                new_position.y = y
                new_position.z = z
                resp = set_field(field.field, 0, new_position)
                print (resp)
            except rospy.ServiceException as e:
                print ("Service call failed: ", e)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

    def set_robot_orientation(self, x, y, z, w):
        service = self.model_name+'/supervisor/node/get_field'
        if self.verbose: rospy.loginfo("Waiting for service %s", service)
        rospy.wait_for_service(service)
        try:
            request_field = rospy.ServiceProxy(service, node_get_field)
            field = request_field(self.robot_node.value, 'rotation')
            print (field)

            other_service = self.model_name+'/supervisor/field/set_rotation'
            if self.verbose: rospy.loginfo("Waiting for service %s", service)
            rospy.wait_for_service(other_service)
            try:
                set_field = rospy.ServiceProxy(other_service, field_set_rotation)
                new_orientation = Quaternion()
                # remember that the rotation in webots is different from
                # rotation in ROS. Webots x,y,z is in m and w is in rad
                # (yaw)
                new_orientation.x = x
                new_orientation.y = y
                new_orientation.z = z
                new_orientation.w = w

                resp = set_field(field.field, 0, new_orientation)
                print (resp)
            except rospy.ServiceException as e:
                print ("Service call failed: ", e)

        except rospy.ServiceException as e:
            print ("Service call failed: ", e)

