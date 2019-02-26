## Master Thesis [WARNING] In progress
### Francesco Saverio Zuppichini

*write abstract here*


# Traversability Simulation

The following images shows the traversability patches (92x92px) of the krock robot from bottom to top of the map
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/querry_krock_traversability.png)


## Stucture

```
.
├── agent
│   ├── Agent.py
│   ├── callbacks
│   │   ├── AgentCallback.py
│   │   └── RosBagSaver.py
│   ├── __init__.py
│   └── RospyAgent.py
├── data
├── docker
├── docker-compose.yml
├── env
│   ├── conditions
│   │   ├── __init__.py
│   │   └── __pycache__
│   ├── __init__.py
    ├── krock
│   │   ├── Krock.py
│   │   ├── KrockWebotsEnv.py
│   ├── __pycache__
│   │   └── __init__.cpython-36.pyc
│   └── webots
│       ├── utils.py
│       ├── WebotsEnv.py
│       └── WebotsWorld.py
├── example.ipynb
├── gym.ipynb
├── __init__.py
├── log
├── main.py
├── maps
├── notebooks
│   ├── test.bag
│   └── Visualisation.ipynb
├── parser.py
├── protocols
├── README.md
├── requirements.txt
├── start.sh
└── utils
    ├── History.py
    └── webots2ros
        ├── Supervisor.py

```

A notebook version of this file can be found in `example.ipynb`

## Webots + Krock
![alt text](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/krock.jpg)

In order to run the Krock robot using Webots we provide a [gym](https://gym.openai.com/) env. If you know gym, then you are good to go. It follows a **basic example** (API could change)


```python
import rospy
import pprint

from env.webots.krock import Krock, KrockWebotsEnv

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes2.wbt'
N_STEPS = 4

rospy.init_node("traversability_simulation")
# create our env
env = KrockWebotsEnv.from_file(WORLD_PATH)
# spawn the robot at a random location
init_obs = env.reset()

print('Initial observations:')
pprint.pprint(init_obs)

for _ in range(N_STEPS):
#   go forward for a while
    obs, r, done, _ = env.step(env.GO_FORWARD)
    pprint.pprint(obs)
    if done: break
        
env.step(env.STOP)
```

    /home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes2.wbt
    Initial observations:
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.0864517599655996,
                                 'x': -0.0005665364181127191,
                                 'y': 0.9999990928141088,
                                 'z': -0.001222050509005902},
                 'position': {'x': -1.9305127599700225,
                              'y': 4.894810190634873,
                              'z': 0.4957720465714862}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.0871377743096904,
                                 'x': -0.0010066333003785806,
                                 'y': 0.9999972055037256,
                                 'z': -0.00213908254593838},
                 'position': {'x': -1.9305080998872506,
                              'y': 4.894721262905384,
                              'z': 0.43001982885217077}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.0874009554351765,
                                 'x': -0.001169794986078125,
                                 'y': 0.99999620193469,
                                 'z': -0.002495535190095485},
                 'position': {'x': -1.9305061169946305,
                              'y': 4.89468775053254,
                              'z': 0.2652790167551831}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.0307538662210285,
                                 'x': -0.007757598955433146,
                                 'y': 0.9994824236245451,
                                 'z': 0.03122025823166951},
                 'position': {'x': -1.9308797264231408,
                              'y': 4.898186905390734,
                              'z': 0.09551972689143962}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.0624841678134733,
                                 'x': -0.018664711550003446,
                                 'y': 0.9991789285375764,
                                 'z': 0.035959662251694044},
                 'position': {'x': -1.93438636135767,
                              'y': 4.911562850823643,
                              'z': 0.09167117154457312}}}





    ({'sensors': {'position': {'x': -1.9320344080149618,
        'y': 4.912721207208116,
        'z': 0.10446160308469837},
       'orientation': {'x': -0.0013561011618689943,
        'y': 0.9998850952014214,
        'z': 0.015098257637329785,
        'w': -2.0750843149573397},
       'front_cam': None}},
     0,
     False,
     {})



`KrockWebotsEnv` implements `WebotsEnv` that is a generic env for Webots. 

Due to gym interface we can explore the action space and the observation space easily


```python
print(env.action_space)
print(env.observation_space)
```

    Dict(frontal_freq:Box(), lateral_freq:Box())
    Dict(sensors:Dict(front_cam:Tuple(Box(600, 800, 3), Box(600, 800, 3)), orientation:Dict(w:Box(1,), x:Box(1,), y:Box(1,), z:Box(1,)), position:Dict(x:Box(1,), y:Box(1,), z:Box(1,))))


Select a random action is even esier


```python
obs, r, done, _ = env.step(env.action_space.sample())
env.step(env.STOP)
```




    ({'sensors': {'position': {'x': -1.9234729057365247,
        'y': 4.912669606466581,
        'z': 0.1105117153319196},
       'orientation': {'x': 0.00740726304258441,
        'y': 0.9999563198444479,
        'z': 0.005700075207048437,
        'w': -2.071732998847759},
       'front_cam': None}},
     0,
     False,
     {})



If the camera throttle node is activated, `rosrun topic_tools throttle messages /krock/front_camera/image 30` calling `env.render()` will popup a window with the images from the krock's camera

## Agent

The `Agent` class describes something that interacts with the enviroment. It provides a nice interface to adopt.

### Usage


```python
from agent import Agent

class DummyAgent(Agent):
    def act(self, env, *args, **kwargs):
        print('I am moving!')
    
a = DummyAgent()
a()
```

#### Lifecycle

The agent lifecyle is the following


```python
a() # init
a.act(env) # do something
a.die(env) # RIP, clean up (e.g store stuff empty the state)
```

You need to first create the agent class and then call it to correctly initialise everything. You can think of it like a *lazy* loader.

### ROS
A `Rospy` agent is already implemented and can be found at `agent.RospyAgent`. It adds two methods to the basic `Agent`: `init_publishers` and `init_subscribers`. They must be overrided to initialise the subscribers and listeners



```python
from agent import RospyAgent
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy

class MyRospyAgent(RospyAgent):
    def init_publishers(self):
        return {
            'joy': rospy.Publisher('/joy', Joy, queue_size=1),
        }

    def init_subscribers(self):
        return { 'pose': 
                rospy.Subscriber('/pose', PoseStamped, self.callback_pose)
               }

    def callback_pose(self, data):
        # this will triggers the `on_state_change` method in all callbacks
        self.state['pose'] = data

a = MyRospyAgent()
a()
print(a.subscribers)
print(a.publishers)
```

    {'pose': <rospy.topics.Subscriber object at 0x7f514c7fcc50>}
    {'joy': <rospy.topics.Publisher object at 0x7f514c7fcef0>}


### Callbacks
`Agent` implements the protocol `Callbackable` protocol. A callback can be
added by calling the `add_callback` or `add_callbacks`. 

The following interface expose the callback methods for `Agent`:

```python
class AgentCallback():
    def on_state_change(self, agent, key, value):
        pass

    def on_shut_down(self):
        pass
```


```python
from agent.callbacks import AgentCallback

class MyCallback(AgentCallback):
    def on_state_change(self, agent, key, value):
        print(key, value)
```

Each `Agent` has a `state` that is just a `dict` . Every time it is changed it triggers the `on_state_change` event. This is usefull
to record stuff.


```python
a = DummyAgent()
a.add_callback(MyCallback())
a()

a.state['foo'] = 'foo'
```

    foo foo


#### RosBagSaver

A callback to store the topics of an agent into bags file is implemented


```python
from agent.callbacks import RosBagSaver

a = MyRospyAgent()
a.add_callbacks([RosBagSaver(save_dir='/temp/data', topics=['pose'])])
a()
```

The files will be stored everytime `agent.die()` il called. A new thread is always spawned to avoid blocking


```python

```


