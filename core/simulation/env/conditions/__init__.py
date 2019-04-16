import numpy as np

class Condition():
    def __call__(self, env, *args, **kwargs):
        pass


class IfOneFalseOf(Condition):
    def __init__(self, conditions):
        self.conditions = conditions

    def __call__(self, env, *args, **kwargs):
        conds = [cond(env) for cond in self.conditions]

        return (False in conds)


class IsInside(Condition):
    def __init__(self, tol=-.01, offset=None):
        self.tol = tol
        self.offset = offset

    def __call__(self, env, *args, **kwargs):
        world, agent = env, env.agent

        return self.is_inside(world, agent, self.tol, self.offset)

    @staticmethod
    def is_inside(world, agent, tol=-.01, offsets=[0,0,0]):
        if offsets is None: offsets=[0,0,0]
        pose = agent.state['pose']
        pos = pose.pose.position

        x, y, z = pos.x, pos.y, pos.z


        def check_if_inside(to_check, bounds, offset):
            is_inside = True
            lower, upper = bounds
            if to_check - tol <= lower: is_inside = False
            # upper bound
            if offset is not None: to_check += offset
            if to_check + tol >= upper: is_inside =  False

            return is_inside

        # has_not_fall = world.z + z >= tol
        has_not_fall = True

        is_inside_x = check_if_inside(x, world.x, offsets[0])
        # ROS return an upside y
        is_inside_y = check_if_inside(- y, world.y, offsets[1])
        # print('is_inside_x={}, is_inside_y={}, ({}{})'.format(is_inside_x, is_inside_y, x, y))
        return is_inside_x and is_inside_y and has_not_fall

# TODO this is probably wrong, in the end I did not rely on this to understand if the robot was stuck
# in the postprocessing take all the patches and then before feed them to the network I removed the negative ones
# during training
class IsNotStuck(Condition):
    def __init__(self, n_last=10, tol=0.1):
        self.history = []
        self.n_last = n_last
        self.tol = tol

    def __call__(self, env, *args, **kwargs):
        position = env.agent.state['pose'].pose.position

        is_stuck = False

        if len(self.history) >= self.n_last:
            self.history.pop(0)

            self.history.append([position.x, position.y])

            std = np.std(self.history)

            is_stuck = (std < self.tol).any()

        return not is_stuck
