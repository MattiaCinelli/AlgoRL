# Standard Libraries
from pathlib import Path

# Third party libraries
import numpy as np

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