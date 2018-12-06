import bpy
import rosbag_pandas
import rosbag
import os
import glob
import numpy as np
import pandas as pd
import time
import cv2
from geometry_msgs.msg import PoseStamped, Pose

# Define vertices and faces
verts = [(0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0)]
faces = [(0, 1, 2, 3)]

MAP_KEY = 'Grid'
MAP_PATH = '/home/francesco/Documents/Master-Thesis/core/maps/bars1.png'
TEX_NAME = 'Texture'
MAT_NAME = 'Mat'

bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
bpy.context.scene.cycles.device = 'GPU'

krock = bpy.data.objects['Cube']
krock.name = 'krock'
krock.scale = [0.2, 0.2, 0.2]

if MAP_KEY not in bpy.data.objects:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=513, y_subdivisions=513)
    # # Define mesh and object variables
    # mymesh = bpy.data.meshes.new("Map")
    # map = bpy.data.objects.new("Map", mymesh)

    # #Set location and scene of object
    # map.location = [0,0,0]
    # bpy.context.scene.objects.link(map)

    # #Create mesh
    # mymesh.from_pydata(verts,[],faces)
    # mymesh.update(calc_edges=True)

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

print(np.max(image))
map.location = [map.location[0], map.location[1], map.location[2] - np.max(image)]

scale = (513 * 0.02) / 2

map.scale = (scale, scale, scale)
map.location = [map.location[0], map.location[1], map.location[2]]
# map.rotation_euler = [0,0, np.radians(90)]
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

mat = bpy.data.materials.new('bricks')

map.data.materials.append(mat)
mat.use_nodes = True

# create the texture node
tree = mat.node_tree

links = tree.links
text_brick = tree.nodes.new(type='ShaderNodeTexBrick')

text_brick.offset = 0

text_brick.inputs[1].default_value = [0.471, 0.643, 0.694, 1]
text_brick.inputs[2].default_value = [0.471, 0.643, 0.694, 1]
text_brick.inputs[3].default_value = [1, 1, 1, 1]

text_brick.inputs[4].default_value = 1.0
text_brick.inputs[5].default_value = 0.0005
text_brick.inputs[6].default_value = 0
text_brick.inputs[8].default_value = 0.01
text_brick.inputs[9].default_value = 0.01

diff = tree.nodes['Diffuse BSDF']
# connect to our material
links.new(text_brick.outputs[0], diff.inputs[0])

print(mod)
# bpy.ops.object.modifier_add(type="DISPLACE")
print(map.dimensions)

camera = bpy.data.objects['Camera']

BAG_FOLDER = '/home/francesco/Desktop/carino/vaevictis/data/'
files = glob.glob(BAG_FOLDER + '/flat/**.bag')


def file2pose(file):
    for file in files:
        bag = rosbag.Bag(file)
        for topic, msg, t in bag.read_messages(topics=['pose']):
            print(msg)
            position = msg.pose.position
            orientation = msg.pose.orientation

            yield [[position.x, position.y, position.z],
                   [orientation.w, orientation.x, orientation.y, orientation.z]]


# camera.rotation_mode = 'QUATERNION'

camera = bpy.data.cameras['Camera']
camera.lens_unit = 'FOV'
camera.angle = 1.05

scene = bpy.context.scene

scene.cycles.samples = 32
scene.render.resolution_x = 640
scene.render.resolution_y = 480
camera = bpy.data.objects['Camera']


# bag = rosbag.Bag(files[1])
# for topic, msg, t in bag.read_messages(topics=['pose']):
#     msg = msg
#     break

#     for i, (position, orientation) in enumerate(file2pose(file)):
#         camera.location = position
#         camera.rotation_quaternion = orientation

#         print(position, orientation)

#         break
#         bpy.context.scene.render.image_settings.file_format='JPEG'
#         bpy.context.scene.render.filepath = "/home/francesco/Desktop/diocane/{}.jpg".format(i)
#         bpy.ops.render.render(use_viewport = True, write_still=True)
#     break


camera.parent = krock

krock.location = [1.0, 0.33, -0.165]
krock.rotation_mode = 'QUATERNION'
krock.rotation_quaternion = [-0.267, 0.00000001, 0.99, 0.0001]

camera.location = [0.16, 0, 0]