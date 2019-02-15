def spawn_points2webots_pose(spawn_point, env):
    _, orientation = env.random_position
    x,y = spawn_point * 0.02 - 5
    z = env.get_height(x, y)

    pose = [[x, z + 0.5, y], orientation]

    return pose