
# Masther Thesis
Francesco Saverio Zuppichini

With this project, we collect and estimate ground traversability for a no-wheel crocodile-like robot. We move the robot in a simulated environment recording its pose. Later, we crop for each height map used to generate the ground a patch such as it includes booth the robot in the center and its footprint in the case of maximum advancement.
The original approach is based on 
an already existing methodology that we further expanded with a smaller deep convolutional neural network based on residual connection and the squeeze and excitation operator.
Then, we evaluate the results by visualizing different datasets and custom patches using GRAD-CAM to highlight and discover the strength and weakness of the model.

[WARNING] In progress
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/querry_krock_traversability.png)
*Map traversability from bottom to top (up-hill) for the Krock robot. The brighter the blue the more traversable *
![image](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Master-Thesis/master/resources/images/krock.jpg)
*Krock*


## Stucture

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

TODO 