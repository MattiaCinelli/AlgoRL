"""Simple script to run snips of code"""
# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from matplotlib.table import Table
from typing import List, Dict, Tuple, Optional, Set

# Local imports
from ..logs import logging
from .tool_box import create_directory

class MakeGrid():
    def __init__(
        self, grid_row:int = 3, grid_col:int = 4, plot_name:str = 'grid_world',
        terminal_states:Dict = None, walls:List[Tuple] = None, initial_state:tuple = (0,0),
        images_dir:str = 'images', some_value:float = 0.0,
        ):
        """
        Initializes the grid world
        Args:
        -------------------
        - grid_row: number of rows in the grid
        - grid_col: number of columns in the grid
        - plot_name: str: name of the plot in output
        - terminal_states: dict: {state: value}. The episode will end when the agent reaches a terminal state
        - walls: list: list of tuples of walls. State the agent can't move to
        - initial_state: tuple: initial state, where the agent starts
        - images_dir: str: path to the directory where the images will be saved
        - some_value: float: some value to initialize the grid
        """
        if terminal_states is None:
            terminal_states = {(0, 3): 1, (1, 3): -10}

        # Initialize the grid environment
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.grid = np.zeros((self.grid_row, self.grid_col)) + some_value
        self.plot_name = plot_name
        self.width = 1.0/self.grid_col
        self.height = 1.0/self.grid_row

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

        # Agent position and reward set up
        self.agent_state = initial_state
        self.initial_state = initial_state
        self.images_dir = images_dir
        create_directory(directory_path = self.images_dir)
        self.logger = logging.getLogger("Grid environment initialized")

    def reset(self):
        """
        Resets the environment to the initial state
        """
        self.logger.debug("Resetting environment to initial state")
        self.agent_state = self.initial_state
        self.grid = np.zeros((self.grid_row, self.grid_col))
        return self.agent_state

    def render_state_value(self):
        """
        Renders the grid world in the terminal
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

    def _drew_grid(self, tb, ax):
        for i in range(self.grid_row):
            tb.add_cell(
                i,-1, self.width, self.height, text=i, 
                loc='right', edgecolor='none', facecolor='none',)

        for i in range(self.grid_col):
            tb.add_cell(
                self.grid_row, i, self.width, 1/8, 
                text=i, loc='center', edgecolor='none', facecolor='none',)
        ax.add_table(tb)

    def drew_statevalue_and_policy(self):
        fig, (st_value, policy) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Dynamic Programming')

        st_value.set_title('State values')
        policy.set_title('Best Policy')

        st_value.set_axis_off()
        policy.set_axis_off()
        tb_st_value = Table(st_value, bbox=[0, 0, 1, 1])
        tb_policy = Table(policy, bbox=[0, 0, 1, 1])

        # Add cells
        for (i, j), val in np.ndenumerate(self.grid):
            # State value
            tb_st_value = self.__state_value_sub_method(tb_st_value, i, j, val)

            # Policy
            tb_policy = self.__state_value_sub_policy(tb_policy, i, j, val)

        self._drew_grid(tb_st_value, st_value)
        self._drew_grid(tb_policy, policy)
        
        plt.savefig(Path(self.images_dir, f'{self.plot_name}_ST_and_policy.png'), dpi=300)       

    def draw_state_value(self):
        _, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title('State values')
        tb = Table(ax, bbox=[0, 0, 1, 1])
        # Add cells
        for (i, j), val in np.ndenumerate(self.grid):
            tb = self.__state_value_sub_method(tb, i, j, val)

        self._drew_grid(tb, ax)
        plt.savefig(Path(self.images_dir, f'{self.plot_name}_state_values.png'), dpi=300)

    def drew_policy(self):
        _, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title('Best Policy')
        tb = Table(ax, bbox=[0, 0, 1, 1])

        # Add cells
        for (i, j), val in np.ndenumerate(self.grid):
            tb = self.__state_value_sub_policy(tb, i, j, val)
        self._drew_grid(tb, ax)
        plt.savefig(Path(self.images_dir, f'{self.plot_name}_policy.png'), dpi=300)

    def __state_value_sub_method(self, tb_st_value, i, j, val):
        if np.isnan(val):
            tb_st_value.add_cell(i, j, self.width, self.height, loc='center', facecolor='dimgray')
        elif (i, j) in self.terminal_states_list:
            if self.grid[i, j]>=0:
                tb_st_value.add_cell(i, j, self.width, self.height, text=val, loc='center', facecolor='lightgreen')
            else:
                tb_st_value.add_cell(i, j, self.width, self.height, text=np.round(val, 2), loc='center', facecolor='tomato')
        else:
            tb_st_value.add_cell(i, j, self.width, self.height, text=np.round(val, 2), loc='center', facecolor='white')
        return tb_st_value

    def __state_value_sub_policy(self, tb_policy, i, j, val):
        exploration = [
            self.grid[self.new_state_given_action((i, j), action)]
            for action in self.possible_actions
        ]
        best_actions = [self.possible_actions[x] for x in np.where(np.array(exploration)==max(exploration))[0]]

        if np.isnan(val):
            tb_policy.add_cell(i, j, self.width, self.height, loc='center', facecolor='dimgray')
        elif (i, j) in self.terminal_states_list:
            if self.grid[i, j]>=0:
                tb_policy.add_cell(i, j, self.width, self.height, text=val, loc='center', facecolor='lightgreen')
            else:
                tb_policy.add_cell(i, j, self.width, self.height, text=np.round(val, 2), loc='center', facecolor='tomato')
        else:
            arrows = "$"
            arrow_symbols = {'U':'\u2191', 'D':'\u2193', 'L':'\u2190', 'R':'\u2192'}
            for best in best_actions:
                arrows += arrow_symbols.get(best)
            arrows += "$"
            tb_policy.add_cell(i, j, self.width, self.height, text=arrows, loc='center', facecolor='white')
        return tb_policy

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
