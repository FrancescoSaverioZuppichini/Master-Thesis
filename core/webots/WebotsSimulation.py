from simulation import Simulation
from utils.webots import Supervisor


class WebotsSimulation(Simulation, Supervisor):
    def on_start(self, sim, world, agent, *args, **kwargs):
        agent.spawn()

    def loop(self, world, agent, *args, **kwargs):
        # TODO simulation should be agent agnostic
        agent.act(world)
        agent.sleep()

    def on_finish(self, sim, world, agent, *args, **kwargs):
        agent.stop()
