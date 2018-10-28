from .Simulation  import Simulation


class BasicSimulation(Simulation):
    """
    Basic simulation. It:
    1) spawns the agent at start
    2) calls the agent.act function at every iteration
    3) kills the agent when it is finished
    """
    def on_start(self, sim, world, agent, *args, **kwargs):
        agent.spawn(world)

    def loop(self, world, agent, *args, **kwargs):
        agent.act(world)
        agent.sleep()

    def on_finish(self, sim, world, agent, *args, **kwargs):
        agent.stop()
        agent.die()
        world.die()
