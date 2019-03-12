import bpy
import cv2
import numpy as np

bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.render.engine = 'CYCLES'

def make_tex(tex_path, name):
    tex = bpy.data.textures.new(name, 'IMAGE')
    img = bpy.data.images.load(tex_path)
    tex.image = img

    return tex

class HeightMap():

    def __init__(self, hm_path):
        self.hm_path = hm_path
        self.hm = cv2.imread(hm_path)
        self.hm = cv2.cvtColor(self.hm, cv2.COLOR_BGR2GRAY)
        self.grid = None

    def spawn(self, location, name=None):
        _ = bpy.ops.mesh.primitive_grid_add(x_subdivisions=self.hm.shape[0],
                                            y_subdivisions=self.hm.shape[1],
                                            location=location)

        self.grid = bpy.context.selected_objects[0]

        if name is not None: self.grid.name = name

    def add_disp(self):
        mod = self.grid.modifiers.new("disp", 'DISPLACE')
        tex = make_tex(self.hm_path, 'hm')
        mod.texture = tex
        mod.mid_level = 0

    def add_texture(self, tex_path):
        tex = make_tex(tex_path, 'traversability')

        mat_name = 'traversability'
        mat = bpy.data.materials.new(mat_name)
        slot = mat.texture_slots.add()
        slot.texture = tex
        mat.active_texture = tex

        mat.use_nodes = True
        tree = mat.node_tree

        diff = tree.nodes['Diffuse BSDF']

        color_ramp = tree.nodes.new('ShaderNodeValToRGB')
        # color_ramp.elements[0].color = [0, 0, 255, 0.5]

        img_tex = tree.nodes.new('ShaderNodeTexImage')
        img_tex.image  = bpy.data.images.load(tex_path)

        uv_map = tree.nodes.new('ShaderNodeUVMap')

        links = tree.links

        links.new(color_ramp.outputs[0], diff.inputs[0])
        links.new(img_tex.outputs[0], color_ramp.inputs[0])
        links.new(uv_map.outputs[0], img_tex.inputs[0])

        self.grid.data.materials.append(mat)

        self.unwrap() # need to unwrap the obj to properly link the texture to the grid

    def unwrap(self):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap()
        bpy.ops.object.mode_set(mode='OBJECT')

    def __call__(self, tex_path, location, name=None):
        self.spawn(location, name)
        self.add_disp()
        self.add_texture(tex_path)

        return self.grid


def heightmap_grid(hms):
    heightmaps = []

    location = np.array([0,0,0])

    for hm in hms:
        heightmap = HeightMap(hm)
        heightmaps.append(heightmap)

        heightmap('/home/francesco/Desktop/textures/bars1-0.png',
                  location.tolist())
        location[0] += 2





heightmap_grid(['/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png', '/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png'])

# HeightMap('/home/francesco/Documents/Master-Thesis/core/maps/train/bars1.png')('/home/francesco/Desktop/textures/bars1-0.png')