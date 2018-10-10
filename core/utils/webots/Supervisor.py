#!/usr/bin/env python
#
# implementation of service calls using the supervisor API of webots
#
import sys
import rospy
import time
import numpy as np
import math
from geometry_msgs.msg import Quaternion, Vector3
from webots_ros.srv import *

"""
Little abstraction for easy ros - webots manipulation
Usage:

# create a Node

>>> node = Node.from_def('EL_GRID')

# get a field from that node

>>> x_dim = node['height']

# get an element in that field

>>> el = x_dim[0]
>>> print(el)
>>> print(el.value)
>>> print(type(el.value))
"""


class Supervisor:
    """
    Interface to access all the services exposed by the Webots Supervisor class.
    This class can be inherited to automatically access all the methods,
    or it can be instantiated to be used and shared/
    """
    name = None

    @staticmethod
    def get_service(service, type):
        res = None
        try:
            res = rospy.ServiceProxy(service, type)
        except rospy.ServiceException as e:
            rospy.logerr(e)
        finally:
            return res

    def get_world_node(self):
        service = self.name + '/supervisor/get_root'
        world = self.get_service(service, get_uint64)
        self.world = world()
        return self.world

    def load_world(self, world_name):
        service = self.name + '/supervisor/world_load'
        set_world = self.get_service(service, set_string)
        res = set_world(world_name)
        return res

    def reload_world(self):
        service = self.name + '/supervisor/world_reload'
        res = self.get_service(service, get_uint64)
        return res

    def reset_simulation(self):
        service = self.name + '/supervisor/simulation_reset'
        res = self.get_service(service, get_uint64)
        return res

    def reset_simulation_physics(self):
        service = self.name + '/supervisor/simulation_reset_physics'
        res = self.get_service(service, get_uint64)
        return res

    def get_robot_node(self):
        service = self.name + '/supervisor/get_self'
        res = self.get_service(service, get_uint64)
        self.robot_node = res()
        return self.robot_node

    def get_robot_position(self):
        service = self.name + '/supervisor/node/get_position'
        req_pos = self.get_service(service, node_get_position)
        pos = req_pos(self.robot_node.value)
        return pos

    def get_robot_orientation(self):
        service = self.name + '/supervisor/node/get_orientation'
        req_orient = self.get_service(service, node_get_orientation)
        orient = req_orient(self.robot_node.value)
        return orient

    def set_robot_position(self, x, y, z):
        service = self.name + '/supervisor/node/get_field'

        req_field = self.get_service(service, node_get_field)
        field = req_field(self.robot_node.value, 'translation')

        service = self.name + '/supervisor/field/set_vec3f'
        set_pos = self.get_service(service, field_set_vec3f)
        pos_res = set_pos(field.field, 0, Vector3(x, y, z))

        return pos_res

    def set_robot_orientation(self, x, y, z, w):
        service = self.name + '/supervisor/node/get_field'

        req_field = self.get_service(service, node_get_field)
        field = req_field(self.robot_node.value, 'rotation')

        service = self.name + '/supervisor/field/set_rotation'
        set_rot = self.get_service(service, field_set_rotation)
        rot_res = set_rot(field.field, 0, Quaternion(x, y, x, w))

        return rot_res

    def restart_robot(self):
        service = self.name + '/krock/supervisor/node/restart_controller'

    def enable_front_camera(self, enable=1):
        service = self.name + '/front_camera/enable'

        request_enable = self.get_service(service, set_int)
        res = request_enable(enable)

        return res


class Field(Supervisor):
    WEBOTS_TYPE_TO_ROS = {
        'SFInt32': field_get_int32,
        'MFFloat': field_get_float,
        'SFBool': field_get_bool,
        'SFRotation': field_get_rotation,
        'SFVec3f': field_get_vec3f,
        'SFFloat': field_get_float
    }

    WEBOTS_TYPE_TO_URL = {
        'SFInt32': 'get_int32',
        'MFFloat': 'get_float',
        'SFBool': 'get_bool',
        'SFRotation': 'get_rotation',
        'SFFloat': 'get_float',
        'SFVec3f': 'get_vec3f'
    }

    def __init__(self, field, node):
        super().__init__()
        self.field = field
        self.node = node

    @property
    def get_type(self):
        service = self.node.name + '/supervisor/field/get_type_name'
        req = self.get_service(service, field_get_type_name)
        type_name = req(self.field.field)

        return type_name, self.WEBOTS_TYPE_TO_ROS[type_name.name]

    # Not used now
    def get_type_url(self, webots_type: str):
        type = "".join(webots_type.split('SF')[-1]).lower()

        return 'get_' + type

    def __getitem__(self, index):
        wb_type, ros_type = self.get_type
        # url = self.get_type_url(wb_type.name)
        url = self.WEBOTS_TYPE_TO_URL[wb_type.name]
        service = self.node.name + '/supervisor/field/{}'.format(url)
        req = self.get_service(service, ros_type)
        value = req(self.field.field, index)
        return value


class Node(Supervisor):
    def __init__(self, node):
        super().__init__()
        self.node = node

    @classmethod
    def from_def(cls, base: str, name: str):
        service = base + '/supervisor/get_from_def'
        req = cls.get_service(service, supervisor_get_from_def)
        node = req(name)

        node = cls(node)
        node.name = base

        return node

    def __getitem__(self, key):
        service = self.name + '/supervisor/node/get_field'
        req = self.get_service(service, node_get_field)
        wb_field = req(self.node.node, key)

        field = Field(wb_field, self)
        field.name = self.name

        return field
