from simulation.env.controller import Controller

class FixedController(Controller):
    def __call__(self, env):
        return env.GO_FORWARD