# Standard Libraries
import os
import numpy as np
import pandas as pd

# Third party libraries
import gym
from icecream import ic
# Local imports
# os.chdir(os.path.dirname(__file__))

from algorl.logs import logging
# from algorl.src.grid_environment import *
from algorl.src.DP import DP, DPv2


def run_DP(env, grid_name):
    ic(grid_name)
    model = DP(
        env=env
        )

    # env.draw_state_value()
    # env.drew_policy()
    # env.render_state_value()

# def run_DPv2(env, grid_name):
#     ic(grid_name)
#     model = DPv2(env=env, grid_name=grid_name)
    
    # model.eval_policy(env=env.env)


if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main.py")
    for gym_env_name in ['FrozenLake-v1']:
        env = gym.make(gym_env_name)
        # model = DPv2(env=env, gym_env_name=gym_env_name)
        ic(np.ones([env.nS, env.nA]) / env.nA)
        env.reset()
        ic(env.step(0))
        ic(env.step(1))
        ic(env.step(2))
        ic(env.step(3))
        # env.render()
        # ic(env.action_space)
        # ic(env.observation_space)
        # ic(env.P.keys())
        # ic(env.ncol)
        # ic(env.nrow)
        # ic(env.s)
        # ic(env.desc)
        # # ic(dir(env))
        # ic(env.action_space)
        # ic(env.class_name)
        # ic(env.close)
        # ic(env.compute_reward)
        # ic(env.env)
        # ic(env.metadata)
        # ic(env.observation_space)
        # ic(env.reward_range)
        # ic(env.seed)
        # ic(env.spec)
        # ic(env.step)
        

    # for gridword in GridWorldExamples.__subclasses__():
    #     run_DP(gridword.gridword(), grid_name = gridword.__name__)
