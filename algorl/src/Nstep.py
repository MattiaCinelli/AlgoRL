# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import pandas as pd
import numpy as np
from icecream import ic

# Local imports
from ..logs import logging

class NstepFunctions(object):
    """
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        
    def get_action(self):
        '''Return random action among the option fo that state'''
        return np.random.choice(self.env.possible_actions)

    def compute_policy(self, state, action):
        '''Return the state and action given in input plus the reward obtained from the new state'''
        return state, action, self.env.grid[self.env.new_state_given_action(state, action)]

    def compute_trajectory(self):
        '''Create the past S, R from starting state to the terminal state'''
        state = self.starting_state
        action = self.get_action()
        states, actions, rewards = [], [], []
        while not self.env.is_terminal_state(state):            
            new_state = self.env.new_state_given_action(state, action)
            reward = self.env.grid[self.env.new_state_given_action(state, action)] - 1
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = new_state
            action = self.get_action()
        return pd.DataFrame({'state': states, 'action': actions, 'reward': rewards}, index=range(1, len(states)+1))

    def compute_n_step_returns(self, df):
        '''Compute the n-step returns from the trajectory'''
        horizon = self.n_step if self.n_step < df.shape[0] else df.shape[0]
        discounts = np.logspace(0, horizon+1, num=horizon+1, base=self.gamma, endpoint=False)[-horizon:]
        Gt = []
        for index in reversed(df.index):
            nstep_df = df.loc[index+1:index+horizon,:]
            Gt.append(np.sum(nstep_df.reward * discounts[:len(nstep_df.reward)]))
        df['G'] = list(reversed(Gt))
        return df


class NstepTD(NstepFunctions):
    '''
    n-step TD for estimating vi=vi*
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 144.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9, starting_state:tuple=None,
        num_of_epochs:int = 1_000, n_step = 1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - gamma: discount_factor, float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        NstepFunctions.__init__(self)
        self.env = env
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.num_of_epochs = num_of_epochs
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/100) == 0:
                self.logger.info(f'Epoch {epoch}')
                self.env.render_state_value()

            ################
            ## Create Path
            ###############
            trajectory_df = self.compute_trajectory()

            ################
            ## Compute return
            ################
            states_returns = self.compute_n_step_returns(trajectory_df)

            ################
            ## Update state value
            ################
            for index in states_returns.index:
                self.env.grid[states_returns.loc[index].state] =\
                    self.env.grid[states_returns.loc[index].state] +\
                        self.alpha * (states_returns.loc[index, 'G']- self.env.grid[states_returns.loc[index].state])
            

class NstepSARSA():
    '''
    n-step Sarsa for estimating Q==q* or q_{pi}
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 147.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9, starting_state:tuple=None,
        num_of_epochs:int = 1_000, n_step = 1):
        """
        - env: grid_environment: A tabular environment created by Make class
        - gamma: discount_factor, float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        NstepFunctions.__init__(self)
        self.env = env
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.num_of_epochs = num_of_epochs
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/100) == 0:
                self.logger.info(f'Epoch {epoch}')
                self.env.render_state_value()

            ################
            ## Create Path
            ###############
            trajectory_df = self.compute_trajectory()
            ic(trajectory_df)
            sys.exit()

class NstepOffPolicuSARSA():
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 149.
    '''
    pass