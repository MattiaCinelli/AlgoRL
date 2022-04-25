# Standard Libraries
import os
import numpy as np
import pandas as pd

# Third party libraries
import gym
# Local imports
os.chdir(os.path.dirname(__file__))

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

def run_DP_gym(env, grid_name):
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
    for gym_env in ['FrozenLake-v1']:
        run_DP_gym(gym.make(gym_env), gym_env)
    sys.exit()
    for gridword in GridWorldExamples.__subclasses__():
        run_DP(gridword.gridword(), grid_name = gridword.__name__)
