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
from .tool_box import create_directory

class make():
    def __init__(
        self, grid_row:int = 3, grid_col:int = 4, 
        terminal_states:Dict = None, walls:List[Tuple] = None, initial_state:tuple = (0,0),
        images_dir:str = 'images', plot_name:str = 'grid_world'
        ):
        """
        Initializes the grid world
        """
        if terminal_states is None:
            terminal_states = {(0, 3): 1, (1, 3): -10}

        # Initialize the grid environment
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.grid = np.zeros((self.grid_row, self.grid_col))

        # States set up
        ## A list of all possible states
        self.all_states = list(product(*[range(self.grid_row),range(self.grid_col)]))
        ## A list of all states the agent can be in
        self.available_states = list(product(*[range(self.grid_row),range(self.grid_col)]))

        # Adding terminal states
        self.terminal_states_list = []
        for key in terminal_states:
            self.terminal_states_list.append(key)
            self.available_states.remove(key)
            self.grid[key] = terminal_states[key]

        # Adding walls
        self.wall_states_list = []
        if walls is not None:
            for wall in walls:
                self.wall_states_list.append(wall)
                self.available_states.remove(wall)
                self.grid[wall] = np.nan

        # Action set up
        self.possible_actions = ['U', 'D', 'L', 'R']
        self.action_space = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}
        # self.action_prob = 0.25

        # Agent position and reward set up
        self.agent_state = initial_state
        self.initial_state = initial_state
        self.images_dir = images_dir
        create_directory(directory_path = self.images_dir)
        self.plot_name = plot_name


    def reset(self):
        """
        Resets the environment to the initial state
        """
        self.agent_state = self.initial_state
        self.grid = np.zeros((self.grid_row, self.grid_col))
        return self.agent_state


    def render_state_value(self):
        """
        Renders the grid world
        """
        for row in range(len(self.grid)):
            print("--------"*self.grid_col)
            for col in range(len(self.grid[row])):
                value = self.grid[row, col]
                if col == 0:
                    print("| {0:.2f} |".format(value), end="")
                else:
                    print(" {0:.2f} |".format(value), end="")
            print("")
        print("--------"*self.grid_col)


    def _drew_grid(self, tb, width, height, ax):
        for i in range(self.grid_row):
            tb.add_cell(i,-1, width, height, text=i, loc='right', edgecolor='none', facecolor='none',)

        for i in range(self.grid_col):
            tb.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none',)
        ax.add_table(tb)
        

    def draw_state_value(self):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        width = 1.0/self.grid_col
        height = 1.0/self.grid_row 

        # Add cells
        for (i, j), val in np.ndenumerate(self.grid):
            if np.isnan(val):
                tb.add_cell(i, j, width, height, loc='center', facecolor='dimgray')
            elif (i, j) in self.terminal_states_list:
                if self.grid[i, j]>=0:
                    tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='lightgreen')
                else:
                    tb.add_cell(i, j, width, height, text=np.round(val, 2), loc='center', facecolor='tomato')
            else:
                tb.add_cell(i, j, width, height, text=np.round(val, 2), loc='center', facecolor='white')

        self._drew_grid(tb, width, height, ax)
        plt.savefig(Path(self.images_dir, self.plot_name+'_policy.png'), dpi=300)


    def drew_policy(self):
        arrow_symbols = {'U':'\u2191', 'D':'\u2193', 'L':'\u2190', 'R':'\u2192'}
        fig, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        width = 1.0/self.grid_col
        height = 1.0/self.grid_row
        # Add cells
        for (i, j), val in np.ndenumerate(self.grid):
            exploration = [
                self.grid[self.new_state_given_action((i, j), action)]
                for action in self.possible_actions
            ]

            best_actions = [self.possible_actions[x] for x in np.where(np.array(exploration)==max(exploration))[0]]

            if np.isnan(val):
                tb.add_cell(i, j, width, height, loc='center', facecolor='dimgray')
            elif (i, j) in self.terminal_states_list:
                if self.grid[i, j]>=0:
                    tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='lightgreen')
                else:
                    tb.add_cell(i, j, width, height, text=np.round(val, 2), loc='center', facecolor='tomato')
            else:
                arrows = "$"
                for best in best_actions:
                    arrows += arrow_symbols.get(best)
                arrows += "$"
                tb.add_cell(i, j, width, height, text=arrows, loc='center', facecolor='white')

        self._drew_grid(tb, width, height, ax)
        plt.savefig(Path(self.images_dir, self.plot_name+'_state_values.png'), dpi=300)


    def new_state_given_action(self, state, action):
        """ Given a state and an action, returns the new state """
        new_state = tuple(map(sum, zip(state, self.action_space[action]))) # new state given action
        ## Bump into wall or border
        if new_state in self.wall_states_list or new_state not in self.all_states:
            return state
        ## if new state is terminal state
        elif self.is_terminal_state(new_state):
            return new_state
        ## if new state is a valid state
        else:
            return new_state


    def is_terminal_state(self, state):
        """ Returns true if the state is a terminal state"""
        return state in self.terminal_states_list
