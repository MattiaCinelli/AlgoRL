# Standard Libraries
from pathlib import Path

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Local imports
from ..logs import logging
logger = logging.getLogger("tool box")

def create_directory(directory_path:str) -> None:
    """Create new folder in path"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


class RLFunctions(object):
    """
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_random_action(self, _):
        return np.random.choice(self.env.possible_actions) 
    
    def epsilon_greedy(self, action:str):
        return np.random.choice(self.env.possible_actions) if np.random.rand() < self.epsilon else action

    def greedy(self, state:tuple, use_epsilon=True):
        '''Return the action with the highest state value return'''
        # All actions with highest state value return
        greedy_actions = [self.env.grid[self.env.next_state_given_action(state, action)] for action in self.env.possible_actions]
        # select random index among max values in list
        action = self.env.possible_actions[np.random.choice([idx for idx, val in enumerate(greedy_actions) if val == max(greedy_actions)])]
        if use_epsilon:
            return self.epsilon_greedy(action)
        return action

    def compute_policy(self, state:tuple, action:str):
        '''Return the state and action given in input plus the reward obtained from the new state'''
        return state, action, self.env.grid[self.env.next_state_given_action(state, action)]

    def _drew_grid(self, tb, width, height, ax):
        for i in range(self.env.grid_row):
            tb.add_cell(i,-1, width, height, text=i, loc='right', edgecolor='none', facecolor='none',)

        for i in range(self.env.grid_col):
            tb.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none',)
        ax.add_table(tb)

    def drew_policy(self, df, plot_name:str):
        df = df.idxmax(axis=1)
        self.logger.debug(df)
        arrow_symbols = {'U':'\u2191', 'D':'\u2193', 'L':'\u2190', 'R':'\u2192'}
        _, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        width = 1.0/self.env.grid_col
        height = 1.0/self.env.grid_row
        # Add cells
        for i, j in df.index:
            if (i, j) in self.env.wall_states_list:
                tb.add_cell(i, j, width, height, loc='center', facecolor='dimgray')
            elif (i, j) in self.env.terminal_states_list:
                if self.env.grid[i, j]>=0:
                    tb.add_cell(i, j, width, height, text=self.env.grid[i, j], loc='center', facecolor='lightgreen')
                else:
                    tb.add_cell(i, j, width, height, text=np.round(self.env.grid[i, j], 2), loc='center', facecolor='tomato')
            else:
                arrows = f"${arrow_symbols[df[i, j]]}$"
                tb.add_cell(i, j, width, height, text=arrows, loc='center', facecolor='white')

        self._drew_grid(tb, width, height, ax)
        plt.savefig(Path(self.env.images_dir, f'{plot_name}_state_values.png'), dpi=300)