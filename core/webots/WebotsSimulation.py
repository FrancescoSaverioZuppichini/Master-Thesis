from simulation import Simulation
from utils.webots import Supervisor

class WebotsSimulation(Simulation, Supervisor):
    def on_start(self, sim, world, agent, *args, **kwargs):
        # self.load_world(str(world.path))
        agent.spawn()

    def loop(self, world, agent, *args, **kwargs):
        agent.sleep()
        agent.move(gait=1,
                   frontal_freq=1.0,
                   lateral_freq=0,
                   manual_mode=True)
        agent.sleep()

    def on_finish(self, sim, world, agent, *args, **kwargs):
        agent.stop()
        agent.die()