# Standard Libraries
import sys
import itertools
from pathlib import Path

# Third party libraries
import pandas as pd
import numpy as np
from icecream import ic

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
        # Gt = []
        for index in reversed(df.index):
            nstep_df = df.loc[index+1:index+horizon, : ]
            G = np.sum(nstep_df.reward * discounts[:len(nstep_df.reward)])
            if index + self.n_step < df.shape[0]:
                G += self.gamma * self.env.grid[df.loc[index + self.n_step].state]

            # Update state value
            self.env.grid[df.loc[index].state] +=\
                self.alpha * (G - self.env.grid[df.loc[index].state])
        return df

    def compute_sarsa_nstep_returns(self, df:pd.DataFrame, Q:np.array):
        '''Compute the n-step returns from the trajectory'''
        horizon = self.n_step if self.n_step < df.shape[0] else df.shape[0]
        discounts = np.logspace(0, horizon+1, num=horizon+1, base=self.gamma, endpoint=False)[-horizon:]

        for index in reversed(df.index):
            nstep_df = df.loc[index+1:index+horizon, : ]
            G = np.sum(nstep_df.reward * discounts[:len(nstep_df.reward)])
            if index + self.n_step < df.shape[0]:
                G += self.gamma *\
                    Q[self.env.all_states.index(df.loc[index + self.n_step, 'state'])][self.env.possible_actions.index(df.loc[index + self.n_step, 'action'])]

            # Update state value
            Q[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])] +=\
                self.alpha *\
                    (G - Q[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])])
        return Q

    def compute_n_step_returns(self, df:pd.DataFrame, Q:np.array, E:np.array):
        '''Compute the n-step returns from the trajectory'''
        # Taken by https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_06/chapter-06.ipynb
        # but my implementation my be wrong
        for index in reversed(df.index):
            td_target = df.loc[index, 'reward'] + self.gamma *\
                Q[self.env.all_states.index(df.loc[index, 'next_state'])][self.env.possible_actions.index(df.loc[index, 'next_action'])]
            td_error = td_target -\
                Q[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])]

            E[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])] =\
                E[self.env.all_states.index(df.loc[index, 'state'])][self.env.possible_actions.index(df.loc[index, 'action'])] + 1
            
            Q = Q + self.alpha * td_error * E
        return Q

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
            if epoch % 100 == 0:
                self.logger.info(f'Epoch {epoch}')
                self.env.render_state_value()

            ## 1. Create Path
            trajectory_df = self.compute_trajectory()

            ## 2. Compute return
            self.compute_td_nstep_returns(trajectory_df)


class NStepSARSA(NStepFunctions):
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

    def compute_state_value(self):
        Q = np.zeros((len(self.env.all_states), len(self.env.possible_actions)), dtype=np.float64)
        E = np.zeros((len(self.env.all_states), len(self.env.possible_actions)), dtype=np.float64)
        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'Epoch {epoch}')
                self.env.render_state_value()

            E.fill(0)
            ## 1. Create Path
            trajectory_df = self.compute_trajectory(action_method='greedy')

            ## 2. Compute Q
            # Q = self.compute_n_step_returns(trajectory_df, Q, E)
            Q = self.compute_sarsa_nstep_returns(trajectory_df, Q)

        final_q = pd.DataFrame(
            data=Q,
            index=self.env.all_states,
            columns=self.env.possible_actions)
        
        self.drew_policy(final_q, plot_name='sarsa')
 

class NStepOffPolicySARSA(NStepFunctions):
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 149.
    '''
    pass

class NStepTreeBackup(NStepFunctions):
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 155.
    '''
    pass

class OffPolicyNStepQSigma(NStepFunctions):
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 156.
    '''
    pass