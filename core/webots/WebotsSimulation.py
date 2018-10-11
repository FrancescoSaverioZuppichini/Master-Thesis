from simulation import Simulation
from utils.webots2ros import Supervisor


class WebotsSimulation(Simulation, Supervisor):
    def on_start(self, sim, world, agent, *args, **kwargs):
        agent.spawn(world)

    def loop(self, world, agent, *args, **kwargs):
        agent.act(world)
        agent.sleep()

    def on_finish(self, sim, world, agent, *args, **kwargs):
        agent.stop()
