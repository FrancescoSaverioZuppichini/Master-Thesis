from utilities.patches.texture import get_rocks
from simulation.Simulation import Simulation
from simulation.env.spawn.SpawnStragety import FlatGroundSpawnStrategy, RandomSpawnStrategy
from os import path

MAPS_DIR = './maps/new-train/'
BAGS_DIR = '/media/francesco/saetta/krock-dataset/new-train/bags/'
map_path = lambda x: path.abspath(MAPS_DIR + "/{}.png".format(x))

MAX_TIME = 10
N_SIM = 50

sims = [
    Simulation(map_path('bumps0'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps0-rocks0'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps0-rocks1'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('bumps1'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps1-rocks0'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps1-rocks1'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('bumps2'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps2'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps2-rocks0'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps2-rocks1'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('bumps3'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps3-rocks0'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bumps3-rocks1'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('steps1'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('steps2'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('steps3'), N_SIM, 1, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('rails1'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME * 2),
    Simulation(map_path('rails2'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('rails3'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('bars1'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bars1'), N_SIM, 2, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('bars3'), N_SIM, 1, BAGS_DIR, FlatGroundSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('ramp0-rocks1'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('ramp0-rocks2'), N_SIM, 2, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('ramp1'), N_SIM, 5, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('ramp1'), N_SIM, 4, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('ramp1'), N_SIM, 3, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),

    Simulation(map_path('slope_rocks1'), N_SIM, 5, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('slope_rocks1'), N_SIM, 4, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
    Simulation(map_path('slope_rocks1'), N_SIM, 3, BAGS_DIR, RandomSpawnStrategy, max_time=MAX_TIME),
]

print('[INFO] TOT sims = {}  N maps = {}'.format(N_SIM * len(sims), len(sims)))
for sim in sims:
    sim()
