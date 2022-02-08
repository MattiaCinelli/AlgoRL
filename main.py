"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries
import matplotlib.pyplot as plt

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import make
from algorl.src.tabular import DP

def print_tab_A():
    env = make(walls = [(1, 1)])
    # print(env)
    # print(env.return_grid())
    model = DP(
        env=env, #verbose = 0
        )
    print(model.compute_state_value())
    model.draw_state_value()
    model.drew_policy()




if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main.py")
    print_tab_A()