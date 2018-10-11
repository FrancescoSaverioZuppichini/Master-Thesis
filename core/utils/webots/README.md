# Webots 
## Utilities to easy work with ROS and Webots
Probably you want to avoid all the ceremony around asking a field value to Webots while using ROS. This package contains an easy object oriented interface that hides all the complexity needed to fetch a field value from a node.

### Getting Started
Let's see some code:

```python
from utils.webots import Node 
# create a Node

node = Node.from_def('/krock','EL_GRID')

# get a field from that node

h = node['height']

# get an element in that field
el = h[0]
print(el)
print(el.value)
print(type(el.value))
el = h[1]
...

```
A `Node` istance is used to get a node from Webots. The easier way is to use the builder `from_def`

exposed by ros.
```
node = Node.from_def('EL_GRID')
```
Under the hood the correct service is called used ROS.

`Node` implementes `Supervisor` that is an interface with a wide array of useful methods.

To get a field from a node you
can access it as it was a dictionary since it overrides `__getitem__`
```python
node = Node.from_def('EL_GRID')
# get a field from a node
field = node['some_field_name'] 
```

This will return a `Field` instance that can be also accessed as an array by passing the index of the value we want.

```python
# get an element in that field
field_response = field[0]
field_value = field[0].name
```

Under the hood `Node` is able to correctly infer the type of a field by fist asking webots the
correct type and then using a dictionary to get the correct ROS msg and the url to make
the service request.

### TODO
- change the name in something like **webots2ros**
- create setup.py