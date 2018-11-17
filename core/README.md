
# Traversability Simulation
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

## Webots + Krock
![alt text](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/gym/resources/images/krock.jpg")

In order to run the Krock robot using Webots we provide a [gym](https://gym.openai.com/) env. If you know gym, then you are good to go. It follows a **basic example** (API could change)


```python
import rospy
import pprint

from env.webots.krock import Krock, KrockWebotsEnv

WORLD_PATH = '/home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes2.wbt'
EPISODES = 25

rospy.init_node("traversability_simulation")
# create our env
env = KrockWebotsEnv.from_file(WORLD_PATH)
# spawn the robot at a random location
init_obs = env.reset()

print('Initial observations:')
pprint.pprint(init_obs)

for _ in range(EPISODES):
#   go forward for a while
    obs, r, done, _ = env.step(env.GO_FORWARD)
    pprint.pprint(obs)
    if done: break
        
env.step(env.STOP)
```

    /home/francesco/Documents/Master-Thesis/core/env/webots/krock/krock2_ros/worlds/holes2.wbt
    Initial observations:
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5216734395810323,
                                 'x': 7.173338268206636e-05,
                                 'y': 0.9999999971706306,
                                 'z': 2.2650841092634077e-05},
                 'position': {'x': 2.6889421226713575,
                              'y': -3.6807569913877285,
                              'z': 0.4942669181354657}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5221118243535807,
                                 'x': -6.216395821302278e-05,
                                 'y': 0.9999998443585398,
                                 'z': -0.0005544533690916926},
                 'position': {'x': 2.688969389807841,
                              'y': -3.680806709380935,
                              'z': 0.41071718924640893}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5222632242264305,
                                 'x': -0.00010453976102288602,
                                 'y': 0.9999997090386967,
                                 'z': -0.0007556414231024095},
                 'position': {'x': 2.6889786473884834,
                              'y': -3.680823559224554,
                              'z': 0.22099154491008136}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5246690210191307,
                                 'x': -0.01998966581487207,
                                 'y': 0.9983781592037247,
                                 'z': -0.05330538889823146},
                 'position': {'x': 2.686942088492735,
                              'y': -3.6849640372591432,
                              'z': 0.09675941465553667}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.45960213543958,
                                 'x': 0.040100387540562325,
                                 'y': 0.9988347822223073,
                                 'z': -0.026852127327504777},
                 'position': {'x': 2.6946672435346857,
                              'y': -3.6839983735802946,
                              'z': 0.12806075368077635}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.422118340713101,
                                 'x': 0.02062454205754777,
                                 'y': 0.9997159190376184,
                                 'z': -0.011946107637501115},
                 'position': {'x': 2.700782403463701,
                              'y': -3.6642346606928355,
                              'z': 0.11168675119111833}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.482109513773996,
                                 'x': 0.029171162724129247,
                                 'y': 0.999394851506375,
                                 'z': -0.01894661045868331},
                 'position': {'x': 2.6862832298421084,
                              'y': -3.6498115198681136,
                              'z': 0.11735236393141552}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5606671614852474,
                                 'x': 0.05581791039823515,
                                 'y': 0.9968540883236492,
                                 'z': -0.056269774046113656},
                 'position': {'x': 2.6663638762781834,
                              'y': -3.63949120232523,
                              'z': 0.13412147731199597}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.630402314211892,
                                 'x': 0.07689464271770177,
                                 'y': 0.9928626772014469,
                                 'z': -0.0911642371859302},
                 'position': {'x': 2.643157255406566,
                              'y': -3.6287962197320063,
                              'z': 0.15399790907386124}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.6522828384862724,
                                 'x': 0.07803264188794486,
                                 'y': 0.9924371820609784,
                                 'z': -0.09475941358436074},
                 'position': {'x': 2.616361962158913,
                              'y': -3.608854676005785,
                              'z': 0.1639654437895951}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.614123145869313,
                                 'x': 0.06773068857210611,
                                 'y': 0.9966333948996201,
                                 'z': -0.04619989173587166},
                 'position': {'x': 2.594805684236837,
                              'y': -3.5753300846198055,
                              'z': 0.1459537393451532}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5546783496395156,
                                 'x': -0.007110707657123482,
                                 'y': 0.9999691952280958,
                                 'z': 0.0033236172296499996},
                 'position': {'x': 2.562989017878421,
                              'y': -3.5424219249384468,
                              'z': 0.10322108105425297}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.522254079474316,
                                 'x': 0.011010768202541387,
                                 'y': 0.9998726273624503,
                                 'z': -0.011553875319584971},
                 'position': {'x': 2.5503005059704735,
                              'y': -3.533331785848343,
                              'z': 0.10357543852418848}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.4348709380110596,
                                 'x': 0.05026215990504604,
                                 'y': 0.9987349135496714,
                                 'z': -0.0015125272263622383},
                 'position': {'x': 2.5316367700712794,
                              'y': -3.513195737808182,
                              'z': 0.11817909113708046}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.3610251074292377,
                                 'x': 0.0866214145296255,
                                 'y': 0.9961702037106713,
                                 'z': 0.011901923538925396},
                 'position': {'x': 2.5100567123384905,
                              'y': -3.4925151447033267,
                              'z': 0.13519241472137797}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.32704310984917,
                                 'x': 0.09276904896810208,
                                 'y': 0.9955657017861356,
                                 'z': 0.015583227542231255},
                 'position': {'x': 2.484102248104814,
                              'y': -3.4723581171946107,
                              'z': 0.14399989540925726}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.344341934928126,
                                 'x': 0.041749974330040907,
                                 'y': 0.9991250794851043,
                                 'z': -0.0024525878831877968},
                 'position': {'x': 2.443529778870629,
                              'y': -3.4556379342596495,
                              'z': 0.12856843869950638}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.3848552795625544,
                                 'x': 0.002289283266331482,
                                 'y': 0.999994457010146,
                                 'z': -0.002417670595811275},
                 'position': {'x': 2.4129019006285954,
                              'y': -3.4401567446571875,
                              'z': 0.10859198609209063}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.4497396249727683,
                                 'x': 0.014104076325239374,
                                 'y': 0.9998974793133222,
                                 'z': -0.002471010699342286},
                 'position': {'x': 2.3955850907228387,
                              'y': -3.423388159630578,
                              'z': 0.11086397564294194}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.540425539158775,
                                 'x': 0.02332196427273216,
                                 'y': 0.9995841359930276,
                                 'z': -0.01695998388955644},
                 'position': {'x': 2.3696367394694056,
                              'y': -3.4037067989049414,
                              'z': 0.11911170442742816}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.627210313709653,
                                 'x': 0.025605260992475204,
                                 'y': 0.9994015436397665,
                                 'z': -0.0232577982612147},
                 'position': {'x': 2.341073881750029,
                              'y': -3.3836258403804553,
                              'z': 0.124872655693099}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.68214440160477,
                                 'x': 0.012073799951991097,
                                 'y': 0.9999082181547343,
                                 'z': -0.006146431594297},
                 'position': {'x': 2.3083462067411276,
                              'y': -3.3570657815394678,
                              'z': 0.12006390887250948}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.671702136437405,
                                 'x': -0.011375831098151538,
                                 'y': 0.9997040910464141,
                                 'z': 0.021501646725058745},
                 'position': {'x': 2.2843681835849727,
                              'y': -3.3294284664613865,
                              'z': 0.10721098535533376}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.6173130217910114,
                                 'x': -0.005118571790742075,
                                 'y': 0.9999171709347624,
                                 'z': 0.011808958152367332},
                 'position': {'x': 2.2692695785751438,
                              'y': -3.3131618299093835,
                              'z': 0.10950778191037137}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.5630404239915188,
                                 'x': -0.008699408791617859,
                                 'y': 0.999948660513162,
                                 'z': -0.0051958276154330275},
                 'position': {'x': 2.2457955731780217,
                              'y': -3.308196542314005,
                              'z': 0.10446079690445878}}}
    {'sensors': {'front_cam': None,
                 'orientation': {'w': -2.477127400103437,
                                 'x': 0.012951646454236476,
                                 'y': 0.9999145587996184,
                                 'z': 0.0017691677955779589},
                 'position': {'x': 2.218881459873234,
                              'y': -3.2897563040036393,
                              'z': 0.11282100490349649}}}





    ({'sensors': {'position': {'x': 2.2028649318189197,
        'y': -3.2719038864493566,
        'z': 0.12060022390564128},
       'orientation': {'x': 0.03263725754272792,
        'y': 0.9994309209404891,
        'z': 0.008523126664270998,
        'w': -2.4154291941539587},
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


Perfect a random action is even esier


```python
obs, r, done, _ = env.step(env.action_space.sample())
env.step(env.STOP)
```




    ({'sensors': {'position': {'x': 2.2800753279422543,
        'y': -3.031360041337963,
        'z': 0.10485295775513392},
       'orientation': {'x': -0.0032172121360495897,
        'y': 0.9994733477624561,
        'z': 0.03229050415493334,
        'w': -1.9883613196437562},
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

    I am moving!


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

    {'pose': <rospy.topics.Subscriber object at 0x7fea4095ab00>}
    {'joy': <rospy.topics.Publisher object at 0x7fea54def400>}


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
