import time

from simulation import Simulation
from simulation.callbacks import Alarm, SimulationCallback
from agent import Agent

class DummyAgent(Agent):
    def move(self, action, *args, **kwargs):
        print(action)

    def act(self, sim, world, *args, **kwargs):
        self.move('forward')
        self.move('left')

class MySimulation(Simulation):
    def on_start(self, sim, *args, **kwargs):
        print('This is going to be call before starting!')

    def loop(self, world, agent, *args, **kwargs):
        agent.act(self, world)
        time.sleep(1)

class Tick(SimulationCallback):
    def __init__(self):
        self.n = 0

    def tick(self, sim, world, agent, *args, **kwargs):
        self.n += 1
        print(self.n)

N_SIM = 2

agent = DummyAgent()

sim = MySimulation()
sim.add_callback(Alarm(stop_after_s=2))
sim.add_callback(Tick())

for _ in range(N_SIM):
    sim(world=None, agent=agent)