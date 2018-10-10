# Webots 
## Utilities to easy work with ROS and Webots

This package contains an easy interface to interact with Webots using ROS.

### Getting Started
Let's see some code:

```python
from utils.webots import Node 
# create a Node

node = Node.from_def('EL_GRID')

# get a field from that node

x_dim = node['height']

# get an element in that field
el = x_dim[0]
print(el)
print(el.value)
print(type(el.value))
```

All the headless of calling for each service is hidden inside Node
The base interface is `Supervisor` that exposes an array of useful methods to call the services
exposed by ros.

Then we have the `Node` that extends `Supervisor`. To get a field from a node you
can access it as it was a dictionary since it overrides `__getitem__`
```python
# get a field from that node

field = node['some_field_name']
```

This will return a `Field` instance that can be also accessed as an array by passing
the index of the value we want

```python
# get an element in that field
field_response = field[0]
field_value = field[0].name
```

Under the hood `Node` is able to correctly infer the type of a field by fist asking webots the
correct type and then using a dictionary to get the correct ROS msg and the url to make
the service request.

### TODO
- change the name
- create setup.py