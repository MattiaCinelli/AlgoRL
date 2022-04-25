# Standard Libraries

# Third party libraries
import itertools
import sys
import pandas as pd
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

# Local imports
from .tool_box import RLFunctions
from ..logs import logging

# 1. Monte Carlo Prediction to estimate state-action values
# 2. On-policy first-visit Monte Carlo Control algorithm
# 3. Off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm

# Source:
# https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf

class MonteCarloFunctions(RLFunctions):
    """
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def state_reward_path(self, state, action):
        '''Create the past S, R from starting state to the terminal state'''
        state_reward = []
        while not self.env.is_terminal_state(state):            
            new_state = self.env.next_state_given_action(state, action)
            reward = self.env.grid[self.env.next_state_given_action(state, action)] - 1
            state_reward.append((state, reward))

            state = new_state
            action = self.get_random_action(state)
        return state_reward

    def state_action_reward_path(self, state, state_action):
        '''Create the past S, A, R from starting state to the terminal state'''
        state_action_reward = []
        num_of_steps = 0
        while not self.env.is_terminal_state(state) and num_of_steps < self.max_step:
            new_state = self.env.next_state_given_action(state, state_action[state])

            if new_state == state:
                # Add penalty for staying in the same state
                reward = -5 + self.env.grid[self.env.next_state_given_action(state, state_action[state])]
            else:
                reward = -1 + self.env.grid[self.env.next_state_given_action(state, state_action[state])]
            state_action_reward.append((state, state_action[state], reward))

            state = new_state
            num_of_steps += 1
        return state_action_reward

    def compute_states_returns(self, state_reward_path):
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
        return states_returns

    def compute_states_action_returns(self, state_action_reward):
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
        return states_actions_returns


class MCPrediction(MonteCarloFunctions):
    '''
    Monte Carlo Prediction to estimate state-action values
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 92
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, starting_state:tuple=None,
        num_of_epochs:int = 1_000, max_step=100):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        MonteCarloFunctions.__init__(self)
        self.env = env
        self.max_step = max_step
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        # Initialize dictionary for final state value
        # {x:num for x in self.env.available_states}
        # state_value = {state:num for num, state in enumerate(self.env.available_states)}
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                print(f'Epoch {epoch}')

            ################
            ## Create Path
            ###############
            # Compute random first state
            first_state = self.starting_state
            # Random  action for first state
            first_action = self.get_random_action(first_state)
            self.logger.debug(f'First state: {first_state}, action: {first_action}')
            # Compute all following states and actions
            state_reward_path = self.state_reward_path(first_state, first_action)

            ################
            ## Compute return
            ################
            states_returns = self.compute_states_returns(state_reward_path)

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


class FirstVisitMCPredictions(MonteCarloFunctions):
    '''
    First-Visit Monte Carlo Prediction, for estimating V = v_pi
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 92
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, starting_state:tuple=None,
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
        MonteCarloFunctions.__init__(self)
        # Agent position and reward set up
        self.num_episodes = num_episodes
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'Epoch {epoch}')

            ################
            ## Create Path
            ################
            # Compute random first state
            first_state = first_state = self.starting_state
            # Random  action for first state
            first_action = self.get_random_action(first_state)
            # Compute all following states and actions
            state_reward_path = self.state_reward_path(first_state, first_action)

            ################
            ## Compute return
            ################
            states_returns = self.compute_states_returns(state_reward_path)

            ################
            ## Update state value
            ################
            first_time_visit_state = set()
            for state, G in states_returns:
                # Ensure that the state is not visited more than once
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


class MCExploringStarts(MonteCarloFunctions):
    '''
    Monte Carlo Exploring Starts to estimating pi = pi*
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 99

    Initialize:
        pi(s) element of A(s) (arbitrarily), for all s element of S
        Q(s, a) element of R (arbitrarily), for all s element of S, a element of A(s) 
        Returns (s, a) empty list, for all s element of S, a element of A(s)

    Loop forever (for each episode):
        Choose S0 element of S, A0 element of A(S0) randomly such that all pairs have probability > 0 
        Generate an episode from S0, A0, following ⇡: S0, A0, R1, . . . , ST-1, AT-11, RT 
        G <- 0
        Loop for each step of episode, t = T-1,T-2,...,0:
        G <- G + Rt+1
        Unless the pair St,At appearsin S0,A0,S1,A1...,St-1,At-1:
            Append G to Returns(St, At)
            Q(St, At) average(Returns(St, At)) 
            pi(St) argmaxa Q(St, a)
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, num_episodes:int = 100, starting_state:tuple=None,
        num_of_epochs:int = 1_000, max_step=100, plot_name:str = 'MCExploringStarts'):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_episodes: int: number of iterations per epoch
        - num_of_epochs: int: number of epochs 
        - plot_name: str: name of the plot in output
        """
        # Agent position and reward set up
        MonteCarloFunctions.__init__(self)
        self.num_episodes = num_episodes
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.plot_name = plot_name
        self.max_step = max_step
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state

    def compute_state_value(self):
        state_list = []
        action_list = []
        for state, action in itertools.product(
            self.env.available_states, self.env.possible_actions):
            state_list.append(state)
            action_list.append(action)
        state_action_value = pd.DataFrame({
            "states":state_list, 'actions':action_list, 'times':0, 'sum_value':0})

        state_action_initialized = {
            state:self.get_random_action(state) for state in self.env.available_states}
        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'Epoch {epoch}')

            ################
            ## Create Path
            ################
            first_state = first_state = self.starting_state
            state_action_reward = self.state_action_reward_path(
                first_state, state_action = state_action_initialized)

            ################
            ## Compute return
            ################
            states_actions_returns = self.compute_states_action_returns(state_action_reward)

            ################
            ## Update state value
            ################

            first_time_visit_state = set()
            for state, action, Gt in states_actions_returns:
                if (state, action) not in first_time_visit_state:
                    first_time_visit_state.add(state)
                    state_action_value.loc[(state_action_value.states==state) &\
                         (state_action_value.actions==action), 'times'] += 1
                    state_action_value.loc[(state_action_value.states==state) &\
                         (state_action_value.actions==action), 'sum_value'] += Gt

            ################
            ## Update state value
            ################
            state_action_value['average'] =\
                 state_action_value.sum_value/state_action_value.times
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

            state_policy = pd.DataFrame({
                'states':self.env.available_states, 'policies':policies, 'averages':averages})
            state_policy.fillna(0, inplace=True)
            for state in self.env.available_states:
                state_action_initialized[state] =\
                    state_policy.loc[state_policy['states'] == state]['policies'].values[0]
                self.env.grid[state] = state_policy[state_policy.states == state].averages.values[0]
            state_action_value.drop('average', axis=1, inplace=True)



class OnPolicyFirstVisitMCControlEstimatingPi(MonteCarloFunctions): #TODO 
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 101
    '''
    def __init__(self) -> None:
        super().__init__()


class OffPolicyMCPredictionEstimatingQ(MonteCarloFunctions): #TODO
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 110
    '''
    def __init__(self) -> None:
        super().__init__()


class OffPolicyMCControlEstimatingPi(MonteCarloFunctions): #TODO
    '''
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 111
    '''
    def __init__(self) -> None:
        super().__init__()