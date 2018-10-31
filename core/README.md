# Core

## Stucture

```
├── agent # Agent package
├── data
├── docker # contains all the docker files
├── docker-compose.yml
├── __init__.py
├── main.py
├── notebooks # contains the notebooks used to test/visualise
├── parser.py
├── protocols # shared protocols used across the code
├── README.md
├── requirements.txt
├── resources # resources, e.g. webots.deb, used by docker
├── services_test.py
├── simulation # Simulation package
├── start.sh
├── test_camera.py
├── test.py
├── utils # utils code
    ...
│   ├── webots2ros # package to easily call webots using ROS
├── webots # package to use Webots in the current framework in cointains our krock
└── world # World package
```

## How to run the simulation
Be sure that webots is running!

### On local machine
Just run

```python
python3 main.py 
```

If you need help
```python
python3 main.py --help
```

### Docker
First build everything, it may take a while
```python
docker-compose build 
```

```python3 docker-compose up
```

## How to create a simulation pipeline
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
## Simulation

### Usage
```
from simulation import *
```
The `simulation` package contains all the code to create and run a simulation. A simulation is something that runs moving the agent around in the world untils it is stopped.

The basic class is `Simulation`. A useful implementation that moves and agent in world is contained into `BasicSimulation`. 

In order to create a simulation you need to subclass `Simulation` and override the `loop` methods that will be called at each iteration

```python
class MySimulation(Simulation):
    def loop(self, world, agent, *args, **kwargs):
        print('looping forever')
        time.sleep(1)
```
Then, to run it
```

sim = MySimulation()
sim(world, agent)
```
Where `world` and `agent` are instances of `world.World` and `agent.Agent`. You need to first create the simulation class and then call it to correctly initialise everything. You can think of it like a *lazy* loader.

#### Callbacks

 `Simulation` implements the `Callbackable` protocol that exposes a simple observer pattern. To create a callback you need to implement `SimulationCallback`

```python
class SimulationCallback():
    def on_start(self, sim, *args, **kwargs):
        pass

    def on_finished(self, sim, *args, **kwargs):
        pass

    def tick(self, sim, *args, **kwargs):
        pass
```
`tick` is called at every iteration.

Example

```python
from simulation.callbacks import Alarm, SimulationCallback

class Tick(SimulationCallback):
    """
    This callback counts the number of the simulation iteration
    """
    def __init__(self):
        self.n = 0

    def tick(self, sim, world, agent, *args, **kwargs):
        self.n += 1
        print(self.n)

```

To **add callback to a simulation** you can use 

```python
sim = MySimulation()
sim.add_callback(Tick())
sim.add_callbacks([Tick()])
```

## Agent

### Usage
```
from agent import Agent

class DummyAgent(Agent):
    def move(self, *args, **kwargs):
        print('I am moving!')

a = DummyAgent()
a()
```
You need to first create the agent class and then call it to correctly initialise everything. You can think of it like a *lazy* loader.
### ROS
A `Rospy` agent is already implemented and can be found at `agent.RospyAgent`. It adds two methods to the basic `Agent`: `init_publishers` and `init_subscribers`. They must be overrided to initialise the subscribers and listeners

Example:

```python
class MyRospyAgent(RospyAgent):
    def init_publishers(self):
        return {
            'joy': rospy.Publisher('/joy', Joy, queue_size=1),
            'spawn': rospy.Publisher(self.SPAWN, PoseStamped, queue_size=1)
        }

    def init_subscribers(self):
        rospy.Subscriber('/pose', PoseStamped, self.callback_pose)

    def callback_pose(self, data):
        # this will triggers the `on_state_change` method in all callbacks
        self.state['pose'] = data
```
It is convinient to return a dictionary from `init_publishers` because it will be store inside the agent. The publishers can be access in any part of the code by `self.subscribers['spawn](...)`


### Callbacks
`Agent` implements the protocol `Callbackable` protocol. A callback can be
added by calling the `add_callback` or `add_callbacks`. 

The following interface expose the callback methods for `Agent`
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
Each `Agent` has a `state` that is just a `dict` . Every time it is changed it triggers the `on_state_change` event. This is usefull
to record stuff.


**BOOTH `Agent` and `Simulation` are callback themself**