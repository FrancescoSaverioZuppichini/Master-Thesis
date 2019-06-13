
# Masther Thesis
Francesco Saverio Zuppichini

*With this project, we estimate ground traversability for a legged crocodile-like robot. We generate different synthetic grounds and let the robot walk on them in a simulated environment to collect its interaction with the terrain. Then, we train a deep convolutional neural network using the data collected through simulation to predict wether a given ground patch can be traverse or not. Later, we highlight the strength and weakness of our method
    by using interpretability techniques to visualise the network's behaviour in interesting scenarios.*

[WARNING] In progress
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/method.png)
*The proposed traversability framework's main blocks in couter-clockwise order. First we generated meaningful synthetic ground, then we let the robot spawn and walk on them in simulated enviroment while storing its interactions. Later, we crop a region of ground, a patch, for each simulation trajectory around the robot according to its locomotion. We label those images using a defined treshold and fit a deep convolutional neural network to predict traversability.*

![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/querry_krock_traversability.png)
*Map traversability from bottom to top (up-hill) for the Krock robot. The greaner the traversable*
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/krock.jpg)
*Krock*


## Stucture
**Deprecated**

```
.
├── blender // Scripts used to creat the 3d visualization in blenders
├── Config.py
├── docker
├── docker-compose-webots.yml
├── docker-compose.yml
├── estimators
│   ├── callbacks.py
│   ├── datasets // Custom dataset to properly preprocess the data in Pytorch
│   ├── inference.py
│   ├── models 
│   ├── README.md
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   ├── visualise.ipynb
│   └── visualise.py
├── example.ipynb
├── images
│   └── bars1_krock.jpg
├── __init__.py
├── maps
├── notebooks
├── nvidia-docker-compose.yml
├── playground.py
├── protocols // proctols shared across different packages
├── README.md
├── requirements.txt
├── simulation
│   ├── agent
│   │   ├── Agent.py
│   │   ├── callbacks
│   │   └── RospyAgent.py
│   ├── block.py
│   ├── env
│   │   ├── conditions
│   │   ├── spawn
│   │   └── webots
│   └── Simulation.py
├── utilities
│   ├── fix_bag_image_path.py
│   ├── History.py
│   ├── patches // API to create custom patches with different obstacles
│   ├── pipeline.py // API to create custom Pipeline
│   ├── postprocessing // Package to post process the data in order to extract all the patches
│   ├── visualisation // Package to visualize a dataset
│   └── webots2ros // API to easily work with ROS in webots
└── visualization.ipynb


```
Each package as its own documentation and examples.

## Results
Final results with a our MicroResNet
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/results.png)
 