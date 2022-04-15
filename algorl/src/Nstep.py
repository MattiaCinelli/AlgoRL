# Standard Libraries
import sys
import itertools
from pathlib import Path

# Third party libraries
import pandas as pd
import numpy as np
from icecream import ic
from matplotlib.table import Table
import matplotlib.pyplot as plt

from algorl.src.TD import TemporalDifferenceFunctions

# Local imports
from ..logs import logging
from .tool_box import RLFunctions

class NStepFunctions(RLFunctions):
    """
    """
    def __init__(self) -> None:
        RLFunctions.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.action_methods = {
            'random': self.get_random_action, 
            'greedy': self.greedy}

    def compute_trajectory(self, action_method:str='random'):
        '''Create the past S, R from starting state to the terminal state'''
        # Get the starting state and first action of new epoch
        state = self.starting_state
        action = self.action_methods[action_method](state)
        states, actions, rewards, next_states, next_actions = [], [], [], [], []
        while not self.env.is_terminal_state(state):            
            next_state = self.env.next_state_given_action(state, action)
            reward = self.env.grid[self.env.next_state_given_action(state, action)] - 1
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            action = self.action_methods[action_method](state)
            next_states.append(state)
            next_actions.append(action)            
        return pd.DataFrame({
            'state': states, 'action': actions, 'reward': rewards,
            'next_state': next_states, 'next_action': next_actions}, 
            index=range(1, len(states)+1))

    def compute_td_nstep_returns(self, df:pd.DataFrame):
        '''Compute the n-step returns from the trajectory'''
        horizon = self.n_step if self.n_step < df.shape[0] else df.shape[0]
        discounts = np.logspace(0, horizon+1, num=horizon+1, base=self.gamma, endpoint=False)[-horizon:]
        Gt = []
        for index in reversed(df.index):
            nstep_df = df.loc[index+1:index+horizon,:] #TODO this is wrong
            Gt.append(np.sum(nstep_df.reward * discounts[:len(nstep_df.reward)]))
        df['G'] = list(reversed(Gt))
        return df
    
    def compute_n_step_returns(self, df:pd.DataFrame, Q:np.array, E:np.array):
        '''Compute the n-step returns from the trajectory'''
        # If n-step is higher that the trajectory length, force n-step to be the trajectory length
        for index in reversed(df.index):
            td_target = df.loc[index, 'reward'] + self.gamma *\
                Q[self.env.all_states.index(df.loc[index, 'next_state'])][self.env.possible_actions.index(df.loc[index, 'next_action'])]
            td_error = td_target -\
                Q[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])]

            E[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])] =\
                E[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])] + 1
            
            Q = Q + self.alpha * td_error * E
        return Q
        # '''

class NStepTD(NStepFunctions):
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
        NStepFunctions.__init__(self)
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

            ## 1. Create Path
            trajectory_df = self.compute_trajectory()

            ## 2. Compute return
            states_returns = self.compute_td_nstep_returns(trajectory_df)

            ## 3. Update state value
            for index in states_returns.index:
                self.env.grid[states_returns.loc[index].state] =\
                    self.env.grid[states_returns.loc[index].state] +\
                        self.alpha * (states_returns.loc[index, 'G']- self.env.grid[states_returns.loc[index].state])
            

class NStepSARSA(NStepFunctions, TemporalDifferenceFunctions):
    '''
    n-step Sarsa for estimating Q==q* or q_{pi}
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 147.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9, starting_state:tuple=None,
        epsilon:float=.1, num_of_epochs:int = 1_000, n_step = 1):
        """
        - env: grid_environment: A tabular environment created by Make class
        - gamma: discount_factor, float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        NStepFunctions.__init__(self)
        self.env = env
        self.alpha = alpha
        self.n_step = n_step
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_of_epochs = num_of_epochs
        self.starting_state = env.initial_state if starting_state is None else starting_state

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

    def compute_state_value(self):
        pi_track = []
        Q_track = np.zeros(
            (self.num_of_epochs, len(self.env.all_states), len(self.env.possible_actions)), 
            dtype=np.float64)
        Q = np.zeros((len(self.env.all_states), len(self.env.possible_actions)), dtype=np.float64)
        E = np.zeros((len(self.env.all_states), len(self.env.possible_actions)), dtype=np.float64)
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/100) == 0:
                self.logger.info(f'Epoch {epoch}')
                # self.env.render_state_value()
            E.fill(0)
            ## 1. Create Path
            trajectory_df = self.compute_trajectory(action_method='greedy')

            ## 2. Compute Q
            Q = self.compute_n_step_returns(trajectory_df, Q, E)
            Q_track[epoch] = Q
            pi_track.append(np.argmax(Q, axis=1))

        final_q = pd.DataFrame(
            data=Q,    # values
            index=self.env.all_states,    # 1st column as index
            columns=self.env.possible_actions)
        
        self.drew_policy(final_q, plot_name='sarsa')
 

class NStepOffPolicuSARSA(NStepFunctions):
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 149.
    '''
    pass