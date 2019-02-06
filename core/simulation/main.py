from simulation import SimulationPipeline
from .parser import parser

if __name__ == '__main__':
    sim_pip = SimulationPipeline()
    print(sim_pip(parser.args))
