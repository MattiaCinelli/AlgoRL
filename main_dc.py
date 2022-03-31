"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np
import pandas as pd
# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import MakeGrid
from algorl.src.DP import DP

def print_tab_A():
    env = MakeGrid(walls = [(1, 1)])
    model = DP(
        env=env,
        plot_name='table_A'
        )
    model.compute_state_value()
    model.draw_state_value()
    model.drew_policy()

def print_tab_B():
    env = MakeGrid(
        grid_row = 5, 
        grid_col = 5, 
        walls = [(1, 1),(1, 3), (3, 1), (3, 3)], 
        terminal_states = {(4, 4): 1, (0, 4): -10}
        )
    model = DP(
        env=env,
        plot_name='table_B',
        step_cost = -1, 
        gamma = 0.5
        )
    model.compute_state_value()
    model.draw_state_value()
    model.drew_policy()

def print_tab_C():
    env = MakeGrid(
        grid_row = 4, 
        grid_col = 4, 
        terminal_states = {(0, 0): 0, (3, 3): 0}, 
        )
    model = DP(
        env=env,
        plot_name='table_C',
        step_cost = -1, 
        gamma = 0.5
        )
    model.compute_state_value()
    model.draw_state_value()
    model.drew_policy()

def print_tab_D():
    env = MakeGrid(
        grid_row = 7, 
        grid_col = 7, 
        walls = [(1, 1),(1, 3), (3, 1), (3, 3)], 
        terminal_states = {(4, 4): 1, (6, 6): 100}, 
        )

    model = DP(
        env=env,
        plot_name='table_D',
        step_cost = -1, 
        gamma = 0.5
        )
    model.compute_state_value()
    model.draw_state_value()
    model.drew_policy()

def print_tab_E():
    df = pd.read_csv('data/maze_1.csv', header=None)
    rows, cols = np.where(df==0)
    env = MakeGrid(
        grid_row=df.shape[0],
        grid_col=df.shape[1],
        walls=list(zip(rows, cols)),
        terminal_states={(5, 4): 10, (0, 9): -10},
    )

    model = DP(
        env=env,
        plot_name='table_E',
        step_cost = -1, 
        gamma = 0.9
        )
    model.compute_state_value()
    model.draw_state_value()
    model.drew_policy()

if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main.py")
    print_tab_A()
    print_tab_B()
    print_tab_C()
    print_tab_D()
    print_tab_E()