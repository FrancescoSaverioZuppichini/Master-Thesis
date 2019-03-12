import bpy


def make_tex(tex_path, name):
    tex = bpy.data.textures.new(name, 'IMAGE')
    img = bpy.data.images.load(tex_path)
    tex.image = img

    return tex