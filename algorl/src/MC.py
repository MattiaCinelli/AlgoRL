# Standard Libraries

# Third party libraries
import sys
import pandas as pd
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

# Local imports
from ..logs import logging
from .tool_box import create_directory


# 1. Monte Carlo Prediction to estimate state-action values
# 2. On-policy first-visit Monte Carlo Control algorithm
# 3. Off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm

# Source:
# https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf

class MonteCarloFunctions(object):
    """
    """
    def __init__(self) -> None:
        pass
    def get_action(self):
        return np.random.choice(self.env.possible_actions) 

    def compute_policy(self, state, action):
        return state, action, self.env.grid[self.env.new_state_given_action(state, action)]

    def state_reward_path(self, state, action):
        state_reward = []

        while not self.env.is_terminal_state(state):            
            new_state = self.env.new_state_given_action(state, action)
            if new_state == state:
                # Add penalty for staying in the same state
                reward = -1
            else:
                reward = self.env.grid[self.env.new_state_given_action(state, action)]
            state_reward.append((state, reward))

            state = new_state
            action = self.get_action()
        return state_reward

    def state_action_reward_path(self, state, state_action):
        state_action_reward = []
        num_of_steps = 0
        while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
            new_state = self.env.new_state_given_action(state, state_action[state])

            if new_state == state:
                # Add penalty for staying in the same state
                reward = -1 + self.env.grid[self.env.new_state_given_action(state, state_action[state])]
            else:
                reward = self.env.grid[self.env.new_state_given_action(state, state_action[state])]
            state_action_reward.append((state, state_action[state], reward))

            state = new_state
            num_of_steps += 1
        return state_action_reward

class MCPrediction(MonteCarloFunctions):
    '''
    Monte Carlo Prediction to estimate state-action values
    '''
    def __init__(self, env, discount_factor:float = 0.9,  num_of_epochs:int = 1_000):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.env = env

    def compute_state_value(self):
        # Initialize dictionary for final state value
        # {x:num for x in self.env.available_states }
        # state_value = {state:num for num, state in enumerate(self.env.available_states)}
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

        ################
        ## Create Path
        ###############
            # compute random first state and random first action for first state
            first_state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            first_action = self.get_action()
            # Compute all following states and actions
            state_reward_path = self.state_reward_path(first_state, first_action)

        ################
        ## Compute return
        ################
            G = 0
            first = True
            states_returns = []
            for state, reward in reversed(state_reward_path):
                # Terminal states have a value of 0 by definition
                if first:
                    first = False
                else:
                    states_returns.append((state, G))
                G = self.discount_factor * G + reward
            states_returns.reverse()

        ################
        ## Update state value
        ################
            for state, G in states_returns:
                state_value.loc[state_value.states==state, 'times'] += 1
                state_value.loc[state_value.states==state, 'sum_value'] += G

        ################
        ## Update state value
        ################
        state_value['average'] = state_value.sum_value/state_value.times
        for grid_state in state_value.states.unique():
            self.env.grid[grid_state] =state_value[state_value.states == grid_state].average 

        self.env.render_state_value()

class FirstVisitMCPredictions(MonteCarloFunctions): # page 92
    '''
    First-Visit Monte Carlo Prediction, for estimating V = v_pi, page 92
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, 
        num_episodes:int = 100, num_of_epochs:int = 1_000, plot_name:str = 'grid_world'
        ):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_episodes: int: number of iterations per epoch
        - num_of_epochs: int: number of epochs 
        - plot_name: str: name of the plot in output
        """
        # Agent position and reward set up
        self.num_episodes = num_episodes
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.env = env

    def compute_state_value(self):
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

            # compute random first state and random first action for first state
            first_state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]]
            first_action = self.get_action()

            ################
            ## Create Path
            ################
            # compute random first state and random first action for first state
            first_state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            first_action = self.get_action()
            # Compute all following states and actions
            state_reward_path = self.state_reward_path(first_state, first_action)

            ################
            ## Compute return
            ################
            G = 0
            first = True
            states_returns = []
            for state, reward in reversed(state_reward_path):
                # Terminal states have a value of 0 by definition
                if first:
                    first = False
                else:
                    states_returns.append((state, G))
                G = self.discount_factor * G + reward
            states_returns.reverse()

            ################
            ## Update state value
            ################
            first_time_visit_state = set()
            for state, G in states_returns:
                if state not in first_time_visit_state:
                    first_time_visit_state.add(state)
                    state_value.loc[state_value.states==state, 'times'] += 1
                    state_value.loc[state_value.states==state, 'sum_value'] += G

        ################
        ## Update state value
        ################
        state_value['average'] = state_value.sum_value/state_value.times
        for grid_state in state_value.states.unique():
            self.env.grid[grid_state] =state_value[state_value.states == grid_state].average 

        self.env.render_state_value()

class MCExploringStarts(MonteCarloFunctions): # page 99
    '''
    Monte Carlo Exploring Starts to estimating pi = pi*, page 99
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, 
        num_episodes:int = 100, num_of_epochs:int = 1_000, plot_name:str = 'MCExploringStarts'
        ):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_episodes: int: number of iterations per epoch
        - num_of_epochs: int: number of epochs 
        - plot_name: str: name of the plot in output
        """
        # Agent position and reward set up
        self.num_episodes = num_episodes
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.plot_name = plot_name
        self.env = env

    def compute_state_value(self):
        state_list = []
        action_list = []
        for state in self.env.available_states:
            for action in self.env.possible_actions:
                state_list.append(state)
                action_list.append(action)
        state_action_value = pd.DataFrame({"states":state_list, 'actions':action_list, 'times':0, 'sum_value':0})

        state_action_initialized = {state:self.get_action() for state in self.env.available_states}
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

            ################
            ## Create Path
            ################
            first_state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] # (2,0)
            state_action_reward = self.state_action_reward_path(first_state, state_action = state_action_initialized)

            ################
            ## Compute return
            ################
            G = 0
            first = True
            states_actions_returns = []
            for state, action, reward in reversed(state_action_reward):
                # Terminal states have a value of 0 by definition
                if first:
                    first = False
                else:
                    states_actions_returns.append((state, action, G))
                G = self.discount_factor * G + reward
            states_actions_returns.reverse()

            ################
            ## Update state value
            ################

            first_time_visit_state = set()
            for state, action, G_ in states_actions_returns:
                if (state, action) not in first_time_visit_state:
                    first_time_visit_state.add(state)
                    state_action_value.loc[(state_action_value.states==state) & (state_action_value.actions==action), 'times'] += 1
                    state_action_value.loc[(state_action_value.states==state) & (state_action_value.actions==action), 'sum_value'] += G_

            ################
            ## Update state value
            ################
            state_action_value['average'] = state_action_value.sum_value/state_action_value.times
            state_action_value.fillna(0, inplace=True)

            ################
            ## Find best action for each state
            ################
            policies, averages = [], []
            for state in self.env.available_states:
                df = state_action_value[state_action_value.states == state]
                sub_df = df[df.average ==  df.average.max()]
                policies.append(sub_df.sample(1).actions.values[0]) 
                if sub_df.sum()['times']==0.0:
                    averages.append(0.0)
                else:
                    averages.append(sub_df.sum()['sum_value']/sub_df.sum()['times'])

            state_policy = pd.DataFrame({'states':self.env.available_states, 'policies':policies, 'averages':averages})
            state_policy.fillna(0, inplace=True)
            for state in self.env.available_states:
                state_action_initialized[state] = state_policy.loc[state_policy['states'] == state]['policies'].values[0]
                self.env.grid[state] = state_policy[state_policy.states == state].averages.values[0]
            state_action_value.drop('average', axis=1, inplace=True)

        self.env.render_state_value()
        self.env.drew_policy(plot_name=self.plot_name)
        self.env.draw_state_value(plot_name=self.plot_name)

class  OnPolicyFirstVisitMCControlEstimatingPi(MonteCarloFunctions): # page 101
    pass

class OffPolicyMCPredictionEstimatingQ(MonteCarloFunctions): # page 110
    pass

class OffPolicyMCControlEstimatingPi(MonteCarloFunctions): # page 111
    pass