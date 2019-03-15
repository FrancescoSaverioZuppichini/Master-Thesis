import cv2
import glob

import numpy as np


def make_locations(size, ncols, start=None):
    temp = [0, 0, 0] if start is None else start
    locations = [temp]
    
    for i in range(ncols):
        temp = temp.copy()
        temp[0] += size
        locations.append(temp)

    return locations


def heightmap_grid(hms, textures, ncols, start):
    heightmaps = []

    locations = make_locations(2.5, ncols, start)
    
    for i, (hm, tex) in enumerate(zip(hms, textures)):
        heightmap = HeightMap(hm)
        heightmaps.append(heightmap)

        heightmap(tex,
                  locations[i])


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

    def spawn(self, location, scale, name=None):
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
        color_ramp.color_ramp.color_mode = 'HSV'
        color_ramp.color_ramp.hue_interpolation = 'FAR'
        color_ramp.color_ramp.elements[0].color = [0.275, 0.078, 0.322, 1.0]
        color_ramp.color_ramp.elements[1].color = [1.000, 0.898, 0.294, 1.0]
        color_ramp.color_ramp.elements[1].position = 1

        img_tex = tree.nodes.new('ShaderNodeTexImage')
        img_tex.image = bpy.data.images.load(tex_path)

        uv_map = tree.nodes.new('ShaderNodeUVMap')

        links = tree.links

        links.new(color_ramp.outputs[0], diff.inputs[0])
        links.new(img_tex.outputs[0], color_ramp.inputs[0])
        links.new(uv_map.outputs[0], img_tex.inputs[0])

        self.grid.data.materials.append(mat)

        self.unwrap()  # need to unwrap the obj to properly link the texture to the grid

    def unwrap(self):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap()
        bpy.ops.object.mode_set(mode='OBJECT')

    def __call__(self, tex_path, location, name=None):
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.render.engine = 'CYCLES'
        self.spawn(location, name)
        self.add_disp()
        self.add_texture(tex_path)


        return self.grid

    @staticmethod
    def run_from_python():
        """
        This can be used in any others python code, it will create a new process
        to run the script in blender.
        :return:
        """
        import subprocess
        from os import path

        base_dir, _ = path.split(__file__)

        subprocess.run(["blender", '--python', '{}/patches.py'.format(base_dir)])


if __name__ == '__main__':
    import bpy

    patches_true = glob.glob('/home/francesco/Desktop/data/test-patches/patches/1-*.png')
    patches_true.sort()
    textures_true = glob.glob('/home/francesco/Desktop/data/test-patches/textures/1-*.png')
    textures_true.sort()
    
    patches_false = glob.glob('/home/francesco/Desktop/data/test-patches/patches/0-*.png')
    patches_false.sort()
    textures_false = glob.glob('/home/francesco/Desktop/data/test-patches/textures/0-*.png')
    textures_false.sort()

    
    print(len(patches_true))


    heightmap_grid(patches_true, textures_true, ncols=len(patches_true), start=None)
    heightmap_grid(patches_false, textures_false, ncols=len(patches_false), start=[0, 2.5, 0])

    filepath = './patches.blend'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    # bpy.ops.wm.quit_blender()
