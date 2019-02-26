import bpy
import cv2


MAP_NAME = 'querry-big-10.png'

MAP_KEY = 'Grid'
MAP_PATH = '/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png'
TEX_NAME = 'querry'
TEXT_PATH = '/home/francesco/Desktop/textures/bars1.png'



map = bpy.data.objects[MAP_KEY]
image = cv2.imread(MAP_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if image.dtype == 'uint8':
    image = image / 256.
if image.dtype == 'uint16':
    image = image / 65536.
if image.dtype == 'uint32':
    image = image / 4294967296.


nodes = bpy.data.materials['traversability'].node_tree.nodes
nodes.get('Image Texture').image = bpy.data.images.load(TEXT_PATH)

map_texture = bpy.data.textures.get('map')
map_texture.image = bpy.data.images.load(MAP_PATH)
