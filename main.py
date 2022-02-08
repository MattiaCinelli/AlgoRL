"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries
import matplotlib.pyplot as plt

# Local imports
from algorl.logs import logging
from algorl.src.grid_enviroment import make

def print_tab_A():
    env = make(walls = [(1, 1)])

    env.compute_state_value()
    env.draw_state_value()
    env.drew_policy()



if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main.py")
    print_tab_A()