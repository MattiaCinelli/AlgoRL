"""Simple script to run snips of code"""
# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.table import Table
from typing import List, Dict, Tuple, Optional, Set

# Local imports
from ..logs import logging
from .grid_environment import Make

class DP():
    def __init__(
        self, env, step_cost:float = -1, gamma:float = 0.5, 
        noise:float = .0, epsilon:float = 1e-4, plot_name:str = 'grid_world'
        ):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - step_cost: float: cost of moving in the environment
        - gamma: float: discount factor
        - noise: float: probability of taking a action that is not the one chosen
        - epsilon: float: threshold for convergence
        - plot_name: str: name of the plot in output
        """
        # Agent position and reward set up
        self.step_cost = step_cost
        self.gamma = gamma
        self.epsilon = epsilon
        self.noise = noise
        self.env = env
        self.plot_name = plot_name

    def prob_of_action(self, action:str):
        """
        Returns the probability of taking an action
        """
        correct = 1 - self.noise
        wrong = self.noise/2
        return {
            'U': {'U':correct, 'L':wrong, 'R':wrong}, 
            'D': {'D':correct, 'L':wrong, 'R':wrong},
            'L': {'L':correct, 'U':wrong, 'D':wrong},
            'R': {'R':correct, 'U':wrong, 'D':wrong},
            }[action]

    def get_sweep(self):
        new_grid = self.env.grid.copy()
        for state in self.env.available_states:
            exploration = []
            
            for action in self.env.possible_actions:
                possible_move = self.prob_of_action(action)

                exploration.append(
                    sum(
                        self.env.grid[self.env.new_state_given_action(state, move)]*possible_move.get(move)
                        for move in possible_move
                        )
                )

            option_results = pd.DataFrame([exploration], columns=self.env.possible_actions)

            new_grid[state] = self.step_cost + self.gamma*\
                option_results[option_results.idxmax(axis=1)[0]][0]

        if np.array_equal(new_grid, self.env.grid):
            return new_grid, True
        elif np.nansum(np.abs(new_grid - self.env.grid)) < self.epsilon :
            return new_grid, True
        else:
            return new_grid, False

    
    def compute_state_value(self):
        done = False
        while not done:
            self.env.grid, done = self.get_sweep()
        return self.env.grid

    def draw_state_value(self):
        self.env.draw_state_value(plot_name=self.plot_name)

    def drew_policy(self):
        self.env.drew_policy(plot_name=self.plot_name)