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
    def __init__(self, tol=-.01):
        self.tol = tol

    def __call__(self, env, *args, **kwargs):
        world, agent = env.world, env.agent

        return self.is_inside(world, agent)

    @staticmethod
    def is_inside(world, agent, tol=-.01):
        pose = agent.state['pose']
        pos = pose.pose.position

        x, y, z = pos.x, pos.y, pos.z

        def check_if_inside(to_check, bounds):
            lower, upper = bounds
            # TODO bad raising in if statement
            if to_check - tol <= lower:
                return False
            # upper bound
            elif to_check + tol >= upper:
                return False

            return True

        has_not_fall = world.z + z >= tol

        is_inside_x = check_if_inside(x, world.x)
        is_inside_y = check_if_inside(y, world.y)

        return is_inside_x and is_inside_y and has_not_fall


class IsNotStuck(Condition):
    def __init__(self, n_last=10, tol=0.1):
        self.history = []
        self.n_last = n_last
        self.tol = tol

    def __call__(self, env, *args, **kwargs):
        position = env.agent.state['pose'].pose.position

        if len(self.history) >= self.n_last: self.history.pop(0)

        self.history.append([position.x, position.y])

        std = np.std(self.history)

        if (std < self.tol).any(): return False
        print(std)
        return True
