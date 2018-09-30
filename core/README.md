# Simulation

### Getting started
Let's see some code

```python
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
```


In order to properly run a simulation you need to create three 
classes:

- `Agent`: defines the robot interface, e.g. how to move it. A ros version
is already implemented `RosAgent1

- `World`: it contains useful information such as
map name and size. TODO

- `Simulation`: a simulation it's something that loops over and over
until certain condition are reached, e.g. time. It must be subclassed in order
to create custom simulation

### Callbacks
`Agent` and `Simulation` implement the same protocol `Callbackable` that is just an easy observer pattern. A callback can be
added by calling the `add_callback`. 

#### AgentCallback
The following interface expose the callback methods for the Agent
```python
class AgentCallback():
    def on_state_change(self, key, value):
        pass
    
    def on_subscribe(self, topic, data):
        pass

    def on_publish(self, topic, data):
        pass
    
    def on_shut_down(self):
        pass
```
Each `Agent` has a `state` that is just a `dict` that every time it is changed it triggers the `on_state_change` event. This is usefull
to record stuff.

#### SimulationCallback
```python
class SimulationCallback():
    def on_start(self, sim, *args, **kwargs):
        pass

    def on_finished(self, sim, *args, **kwargs):
        pass

    def tick(self, sim, *args, **kwargs):
        pass
```
`tick` is called at every iteration

**BOOTH `Agent` and `Simulation` are callback themself**