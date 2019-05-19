
def spawn_points2webots_pose(spawn_point, env, tr=[5,5], orientation=None):
    orientation = env.random_position[1] if orientation is None else orientation
    x,y = spawn_point
    z = env.get_height(x, y)
    pose = [[(x * 0.02) - tr[0], z + 0.5, (y * 0.02) - tr[1]], orientation]

    return pose

from .SpawnStragety import FlatGroundSpawnStrategy, RandomSpawnStrategy