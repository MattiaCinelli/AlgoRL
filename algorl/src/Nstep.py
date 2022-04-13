# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import pandas as pd
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

# Local imports
from ..logs import logging
from .tool_box import create_directory
from matplotlib.table import Table

class MonteCarloFunctions(object):
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

    def compute_trajectory(self, state):
        '''Create the past S, R from starting state to the terminal state'''
        # Random  action for first state
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


    def compute_states_returns(self, state_reward_path):
        G = 0
        first = True
        states_returns = []
        for state, action, reward in reversed(state_reward_path):
            # Terminal states have a value of 0 by definition
            if first:
                first = False
            else:
                states_returns.append((state, action, G))
            G = self.discount_factor * G + reward
        states_returns.reverse()
        return states_returns

    def compute_n_step_returns(self, df):
        ic(df.shape)
        discounts = np.logspace(0, self.n_step+1, num=self.n_step+1, base=self.gamma, endpoint=False)[-self.n_step:]
        Gt = []
        for index in reversed(df.index):
            # if index <= df.shape[0] - self.n_step:
            nstep_df = df.loc[index+1:index+self.n_step,:]
            # ic(len(nstep_df.reward), discounts[:len(nstep_df.reward)])
            Gt.append(np.sum(nstep_df.reward * discounts[:len(nstep_df.reward)]))
                # ic(index)
                # print(np.sum(nstep_df.reward))
            # else:
            #     Gt.append(df.loc[index,'reward'])
        print(len(Gt))
        print(Gt)
        
        ic(discounts)
        ic(self.n_step)
        #     # Terminal states have a value of 0 by definition
        #     if first:
        #         first = False
        #     else:
        #         states_returns.append((state, action, G))
        #     G = self.discount_factor * G + reward
        # states_returns.reverse()
        # return states_returns

class NstepTD(MonteCarloFunctions):
    '''
    Monte Carlo Prediction to estimate state-action values
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 92
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, gamma:float = 0.9, starting_state:tuple=None,
        num_of_epochs:int = 1_000, max_step=100, n_step = 2):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        MonteCarloFunctions.__init__(self)
        self.env = env
        self.max_step = max_step
        self.n_step = n_step
        self.gamma = gamma
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        # Initialize dictionary for final state value
        # {x:num for x in self.env.available_states}
        # state_value = {state:num for num, state in enumerate(self.env.available_states)}
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

            ################
            ## Create Path
            ###############
            # Compute random first state
            first_state = self.starting_state

            self.logger.debug(f'First state: {first_state}')
            # Compute all following states and actions
            trajectory_df = self.compute_trajectory(first_state)
            ic(trajectory_df)
            # ic(reversed(trajectory_df))
            # sys.exit()
            ################
            ## Compute return
            ################
            # states_returns = self.compute_states_returns(trajectory_path)
            states_returns = self.compute_n_step_returns(trajectory_df)
            ic(states_returns)
            sys.exit()
            ################
            ## Update state value
            ################
            for state, Gt in states_returns:
                state_value.loc[state_value.states == state, 'times'] += 1
                state_value.loc[state_value.states == state, 'sum_value'] += Gt

        ################
        ## Update state value
        ################
        state_value['average'] = state_value.sum_value/state_value.times
        for grid_state in state_value.states.unique():
            self.env.grid[grid_state] = state_value[state_value.states == grid_state].average


class NstepSARSA(): # page 147
    '''

    Page 147 of Sutton and Barto.
    '''
    pass

class NstepOffPolicuSARSA(): # page 149
    '''

    Page 149 of Sutton and Barto.
    '''
    pass