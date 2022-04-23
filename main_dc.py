"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np
import pandas as pd
# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import *
from algorl.src.DP import DP

def run_DP(env, grid_name):
    print(grid_name)
    model = DP(
        env=env
        )
    model.simulate()
    # env.draw_state_value()
    # env.drew_policy()
    env.drew_statevalue_and_policy()
    # env.render_state_value()

if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main.py")
    for gridword in GridWorldExamples.__subclasses__():
        run_DP(gridword.gridword(), grid_name = gridword.__name__)
