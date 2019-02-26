import bpy
# import rosbag
import os
import glob
import numpy as np
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped, Pose

# Define vertices and faces
verts = [(0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0)]
faces = [(0, 1, 2, 3)]

MAP_NAME = 'querry-big-10.png'

MAP_KEY = 'Grid'
MAP_PATH = '/home/francesco/Documents/Master-Thesis/core/maps/test/querry-big-10.png'
TEX_NAME = 'querry'
TEXT_PATH = '/home/francesco/Desktop/textures/querry-None.png'
MAT_NAME = 'Mat'

bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
# bpy.context.scene.cycles.device = 'GPU'

top_cam = bpy.data.cameras.new("Camera")
top_cam_ob = bpy.data.objects.new("Camera", top_cam)
bpy.context.scene.objects.link(top_cam_ob)

top_cam_ob.location = [0, 0, 15]
top_cam_ob.rotation_euler = [0, 0, 0]

bpy.context.scene.camera = top_cam_ob


lamp = bpy.data.lamps['Lamp'].type = 'HEMI'

map = bpy.data.objects[MAP_KEY]
image = cv2.imread(MAP_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if image.dtype == 'uint8':
    image = image / 256.
if image.dtype == 'uint16':
    image = image / 65536.
if image.dtype == 'uint32':
    image = image / 4294967296.


if MAP_KEY not in bpy.data.objects:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=image.shape[0], y_subdivisions=image.shape[0])

map.location = [map.location[0], map.location[1], map.location[2] - np.max(image)]

scale = (image.shape[0] * 0.02) / 2

map.scale = (scale, scale, scale)
map.location = [map.location[0], map.location[1], map.location[2]]
# create texture
tex = bpy.data.textures.new(TEX_NAME, 'IMAGE')
image = bpy.data.images.load(MAP_PATH)
tex.image = image

material = bpy.data.materials.new(MAT_NAME)
# add texture to material
slot = bpy.data.materials[MAT_NAME].texture_slots.add()
slot.texture = tex
bpy.data.materials[MAT_NAME].active_texture = tex
# create modifier
mod = map.modifiers.new("disp", 'DISPLACE')
mod.texture = tex
mod.mid_level = 0

mat = bpy.data.materials.new('traversability')

map.data.materials.append(mat)
mat.use_nodes = True

# create the texture node
tree = mat.node_tree

links = tree.links
ramp = tree.nodes.new(type='ShaderNodeValToRGB')
ramp.color = [0, 0, 1, 1]

image_texture = tree.nodes.new(type='ShaderNodeTexImage')

image_texture.image = bpy.data.images.load(TEXT_PATH)
diff = tree.nodes['Diffuse BSDF']
# connect to our material
links.new(ramp.outputs[0], diff.inputs[0])
links.new(ramp.inputs[0], image_texture.outputs[0])