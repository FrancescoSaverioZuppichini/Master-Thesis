#!/usr/bin/env python
#
# implementation of service calls using the supervisor API of webots2ros
#
import sys
import rospy
import time
import numpy as np
import math
from geometry_msgs.msg import Quaternion, Vector3
from webots_ros.srv import *

"""
Little abstraction for easy ros - webots2ros manipulation
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
    verbose = True

    @staticmethod
    def get_service(service, type):
        rospy.wait_for_service(service, timeout=None)
        res = rospy.ServiceProxy(service, type)
        return res

    @staticmethod
    def call_service(url, type, *args, **kwargs):
        res = None
        try:
            service = Supervisor.get_service(url, type)
            res = service(*args, **kwargs)
        except Exception:
            if Supervisor.verbose: rospy.logwarn(url + ' ' +  str(e))

        finally:
            return res

    def retry_service(self, func, *args, **kwargs):
        res = None
        while res == None:
            res = func(*args, **kwargs)
            time.sleep(0.01)

        return res

    def get_world_node(self):
        service = self.name + '/supervisor/get_root'
        # world = self.call_service(service, get_uint64)
        self.world = self.call_service(service, get_uint64)
        return self.world

    def load_world(self, world_name):
        service = self.name + '/supervisor/world_load'
        res = self.call_service(service, set_string, world_name)
        return res


    def reset_simulation(self):
        service = self.name + '/supervisor/simulation_reset'

        res = self.call_service(service, get_bool)

        return res

    def set_simulation_mode(self, mode):
        service = self.name + '/supervisor/simulation_set_mode'

        res = self.call_service(service, set_int, mode)

        return res

    def reset_simulation_physics(self):
        service = self.name + '/supervisor/simulation_reset_physics'

        res = self.call_service(service, get_bool)

        return res

    def reset_node_physics(self, node):
        service = self.name + '/supervisor/node/reset_physics'

        res = self.call_service(service, node_reset_functions, node.value)

        return res

    def get_robot_node(self):
        service = self.name + '/supervisor/get_self'
        self.robot_node = self.call_service(service, get_uint64)

        return self.robot_node

    def get_robot_position(self):
        service = self.name + '/supervisor/node/get_position'
        pos = self.call_service(service, node_get_position, self.robot_node.value)
        return pos

    def get_robot_orientation(self):
        service = self.name + '/supervisor/node/get_orientation'
        orient = self.call_service(service, node_get_orientation, self.robot_node.value)
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
        service = self.name + '/supervisor/node/restart_controller'

        res = self.call_service(service, node_reset_functions, self.robot_node.value)

        return res

    def enable_front_camera(self, enable=1):
        service = self.name + '/front_camera/enable'

        res = self.call_service(service, set_int, enable)

        return res

    def simulation_set_mode(self, mode):
        service = self.name + '/supervisor/supervisor_simulation_set_mode'

        res = self.call_service(service, set_int, mode)

        return res

    def simulation_get_mode(self):
        service = self.name + '/supervisor/supervisor_simulation_get_mode'

        res = self.call_service(service, get_int)

        return res

    def remove_node(self, node):
        service = self.name + '/supervisor/node/remove'

        res = self.call_service(service, node_remove, node)

        return res

    def remove_field(self, field):
        service = self.name + '/supervisor/field/remove'

        res = self.call_service(service, field_remove, field, -1)

        return res

class Field(Supervisor):
    WEBOTS_TYPE_TO_ROS = {
        'SFInt32': field_get_int32,
        'SFString' : field_get_string,
        'MFFloat': field_get_float,
        'SFBool': field_get_bool,
        'SFRotation': field_get_rotation,
        'SFVec3f': field_get_vec3f,
        'SFFloat': field_get_float,
        'MFNode': field_get_node
    }

    WEBOTS_TYPE_TO_URL = {
        'SFInt32': 'get_int32',
        'SFString': 'get_string',
        'MFFloat': 'get_float',
        'SFBool': 'get_bool',
        'SFRotation': 'get_rotation',
        'SFFloat': 'get_float',
        'SFVec3f': 'get_vec3f',
        'MFNode': 'get_node'
    }

    def __init__(self, field, node):
        super().__init__()
        self.id = field.field
        self.field = field
        self.node = node

    @property
    def get_type(self):
        service = self.node.name + '/supervisor/field/get_type_name'
        type_name = self.call_service(service, field_get_type_name, self.field.field)
        # type_name = req(self.field.field)

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

        value = self.call_service(service, ros_type, self.field.field, index)
        # value = req(self.field.field, index)

        return value


class Node(Supervisor):
    def __init__(self, node):
        super().__init__()
        self.id = node.node
        self.node = node

    @classmethod
    def from_def(cls, base: str, name: str):
        service = base + '/supervisor/get_from_def'
        node = cls.call_service(service, supervisor_get_from_def, name)
        # node = req(name)
        node = cls(node)
        node.name = base

        return node

    def __getitem__(self, key):
        service = self.name + '/supervisor/node/get_field'
        wb_field = self.call_service(service, node_get_field, self.node.node, key)
        # wb_field = req(self.node.node, key)

        field = Field(wb_field, self)
        field.name = self.name

        return field

    def __setitem__(self, key, value):
        if isinstance(key, Field): key = key.id

        service = self.name + '/supervisor/field/import_node_from_string'
        res = self.call_service(service, field_import_node_from_string, key, -1, value)
        # res = req(key, -1, value)

        return res

    def __delitem__(self, key):
        self.remove_field(key.id)

