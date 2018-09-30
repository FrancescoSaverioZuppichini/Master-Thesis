import time

from simulation import Simulation
from simulation.callbacks import Alarm
from agent import Agent

class DummyAgent(Agent):
    def move(self, *args, **kwargs):
        print('I am moving!')

class MySimulation(Simulation):
    def on_start(self, sim, *args, **kwargs):
        print('This is going to be call before starting!')

    def loop(self, world, agent, *args, **kwargs):
        agent.move()
        time.sleep(1)

N_SIM = 2

agent = DummyAgent()

sim = MySimulation()
sim.add_callback(Alarm(stop_after_s=2))

for _ in range(N_SIM):
    sim(world=None, agent=agent)