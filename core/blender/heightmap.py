"""

This scripts load the texture generated by the `/estimators/inference.py` file, where each pixel in the texture is > 1 if the
patch in that point is traversable, and apply it to the ground map in blender in order to create a good looking visualisation.

### Usage

Edit the global variable `MAP_PATH` and `TEXT_PATH`, then

```
blender ./traversability.blend --python heightmap.py
```

"""
import cv2
import numpy as np


class BlenderVisualization():

    def draw_heightmap(self, map_path):
        map_texture = bpy.data.textures.get('map')
        map_texture.image = bpy.data.images.load(map_path)

    def apply_texture(self, text_path):
        nodes = bpy.data.materials['traversability'].node_tree.nodes
        nodes.get('Image Texture').image = bpy.data.images.load(text_path)


    def setup_camera(self):
        cam = bpy.data.objects['Camera']
        # cam.rotation_mode = ''
        cam.location = [7, -5, 9]
        cam.rotation_euler = [np.pi / 4, 0, np.pi / 4]

        bpy.data.scenes['Scene'].camera = cam

        camera = bpy.data.cameras['Camera']

        camera.lens_unit = 'FOV'
        camera.angle = 1.74

    def render(self, file_path):
        scene = bpy.context.scene
        scene.cycles.samples = 32
        scene.render.resolution_x = 4096
        scene.render.resolution_y = 2160

        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(use_viewport=True, write_still=True)


    def __call__(self, map_path, text_path, file_path):
        self.draw_heightmap(map_path)
        self.apply_texture(text_path)
        self.setup_camera()
        self.render(file_path)

        bpy.ops.wm.quit_blender()

    @staticmethod
    def run_from_python():
        """
        This can be used in any others python code, it will create a new process
        to run the script in blender and display the rendered image.
        :return:
        """
        import subprocess
        import matplotlib.pyplot as plt
        from os import path

        base_dir, _ = path.split(__file__)

        subprocess.run(["blender", '{}/traversability.blend'.format(base_dir), '--python', '{}/heightmap.py'.format(base_dir)])

        rendered = cv2.imread(FILE_PATH)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        plt.imshow(rendered)
        plt.show()

if __name__ == '__main__':
    import bpy
    import glob
    from os import path
    TEX_DIR = '/home/francesco/Desktop/textures/'
    MAP_NAME = 'querry-big-10'
    texture_paths = glob.glob('{}/{}-*'.format(TEX_DIR, MAP_NAME))
    print(texture_paths)
    MAP_PATH = '/home/francesco/Documents/Master-Thesis/core/maps/test/{}.png'.format(MAP_NAME)

    TEXT_PATH = '/home/francesco/Desktop/textures/querry-270.png'
    OUT_DIR = '/home/francesco/Desktop/'
    b_vis = BlenderVisualization()

    for text_path in texture_paths:
        name = path.splitext(text_path)[0]
        out_path = '{]/{}.png'.format(OUT_DIR, name)

        b_vis(MAP_PATH, text_path, out_path)



