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

MAP_NAME = 'flat'

MAP_KEY = 'Grid'
MAP_PATH = '/home/francesco/Documents/Master-Thesis/core/maps/{}.png'.format(MAP_NAME)
TEX_NAME = 'Texture'
MAT_NAME = 'Mat'

bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
# bpy.context.scene.cycles.device = 'GPU'

top_cam = bpy.data.cameras.new("Camera")
top_cam_ob = bpy.data.objects.new("Camera", top_cam)
bpy.context.scene.objects.link(top_cam_ob)

top_cam_ob.location = [0,0,15]
top_cam_ob.rotation_euler = [0,0,0]

bpy.context.scene.camera = top_cam_ob

print(list(bpy.data.objects))
krock = bpy.data.objects['krock']
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

# bpy.ops.object.modifier_add(type="DISPLACE")

camera = bpy.data.objects['Camera']

BAG_FOLDER = '/home/francesco/Desktop/carino/vaevictis/data/dataset/'
files = glob.glob(BAG_FOLDER + '/{}/*.csv'.format(MAP_NAME))

def msg2pose(msg):
    position = msg.pose.position
    orientation = msg.pose.orientation

    return [[position.x, position.y, position.z],
            [orientation.w, orientation.x, orientation.y, orientation.z]]

def bag2pose(file_path):
    bag = rosbag.Bag(file_path)
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['pose'])):
            yield msg2pose(msg)

def csv2pose(file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        position = row['pose__pose_position_x'], row['pose__pose_position_y'], row['pose__pose_position_z']
        orientation = row['pose__pose_orientation_x'], row['pose__pose_orientation_y'], row['pose__pose_orientation_z'], row['pose__pose_orientation_w']
        advancement = row['advancement']

        advancement = np.clip(advancement, 0, 0.16)

        yield position, orientation, advancement/ 0.16

def pose(file_path):
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.bag':
        pose = bag2pose(file_path)
    elif file_extension == '.csv':
        pose = csv2pose(file_path)
    return pose

camera = bpy.data.cameras['Camera']
camera.lens_unit = 'FOV'
camera.angle = 1.05

scene = bpy.context.scene

scene.cycles.samples = 32
scene.render.resolution_x = 640
scene.render.resolution_y = 480
camera = bpy.data.objects['Camera']

camera.parent = krock

print(files[0])
frame_idx = 0
skip = 10

bpy.context.scene.objects.active = krock

krock_mat = krock.data.materials[0]
krock_mat.use_nodes = True
tree = krock_mat.node_tree
links = tree.links
# text_brick = tree.nodes.new(type='DiffuseBSDF')
diff = tree.nodes['Diffuse BSDF']
# connect to our material


cmap = plt.cm.get_cmap('Spectral')

for file in files:
    for i, (position, orientation, advancement) in enumerate(pose(file)):
        if i % skip == 0:
            krock.location = position
            krock.rotation_quaternion = orientation
            # print(advancement)
            diff.inputs[0].default_value = cmap(advancement)
            diff.inputs[0].keyframe_insert(data_path="default_value", frame=frame_idx)
            krock.keyframe_insert(data_path="location", frame=frame_idx)
            frame_idx += 1

        # print(position, orientation)
        
        # time.sleep(1)
        # break
        # bpy.context.scene.render.filepath = "/home/francesco/Desktop/diocane/{}.jpg".format(i)
        # bpy.ops.render.render(use_viewport = True, write_still=True)

bpy.context.scene.render.image_settings.file_format='JPEG'

bpy.context.scene.render(animation=True)
# bpy.context.scene.render.filepath = "/home/francesco/Desktop/diocane/{}.jpg".format(i)


# camera.parent = krock

# krock.location = [1.0, 0.33, -0.165]
# krock.rotation_mode = 'QUATERNION'
# krock.rotation_quaternion = [0.98, 0.14, -0.0, -0.03]

# camera.location = [0.16, 0, 0]