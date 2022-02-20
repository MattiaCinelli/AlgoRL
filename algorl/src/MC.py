"""Simple script to run snips of code"""
# Standard Libraries
from curses import KEY_C1
from pathlib import Path

# Third party libraries
import string
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

# Local imports
from ..logs import logging
from .tool_box import create_directory
import sys
from collections import defaultdict
from icecream import ic

# 1. Monte Carlo Prediction to estimate state-action values
# 2. On-policy first-visit Monte Carlo Control algorithm
# 3. Off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm

class MCPrediction():
    '''
    Monte Carlo Prediction to estimate state-action values
    '''
    def __init__(self, env, discount_factor:float = 0.9,  num_of_epochs:int = 10_000):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        self.num_of_epochs = num_of_epochs
        self.discount_factor = discount_factor
        self.env = env

    def get_action(self):
        return np.random.choice(self.env.possible_actions) 

    def compute_policy(self, state, action):
        return state, action, self.env.grid[self.env.new_state_given_action(state, action)]

    def compute_state_value(self):
        # Initialize dictionary for final state value
        # {x:num for x in self.env.available_states }
        # state_value = {state:num for num, state in enumerate(self.env.available_states)}
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

            # compute random first state and random first action for first state
            state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            action = self.get_action()

        ################
        ## Create Path
        ###############
            state_reward = []
            while not self.env.is_terminal_state(state):
                state, action, reward = self.compute_policy(state, action)
                state_reward.append((state, reward))

                state = self.env.new_state_given_action(state, action)
                action = self.get_action()

        ################
        ## COmpute return
        ################
            G = 0
            first = True
            states_returns = []
            for state, reward in reversed(state_reward):
                # a terminal state has a value of 0 by definition
                # this is the first state we encounter in the reversed list
                # we'll ignore its return (G) since it doesn't correspond to any move
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


# df = pd.DataFrame.from_records(states_actions_returns, columns=['state', 'action', 'reward'])
   

class FirstVisitMCPredictions():
    '''
    Monte Carlo Prediction to estimate state-action values
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, 
        num_episodes:int = 100, num_of_epochs:int = 10_000, plot_name:str = 'grid_world'
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

    def get_action(self):
        return np.random.choice(self.env.possible_actions) 

    def compute_policy(self, state, action):
        return state, action, self.env.grid[self.env.new_state_given_action(state, action)]

    def compute_state_value(self):
        # Initialize dictionary for final state value
        # {x:num for x in self.env.available_states }
        # state_value = {state:num for num, state in enumerate(self.env.available_states)}
        state_value = pd.DataFrame({"states":self.env.available_states, 'times':0, 'sum_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')

            # compute random first state and random first action for first state
            state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            # state = (2,0)
            action = self.get_action()

            ##########
            ## Create Path
            ##########
            state_action_reward = []
            while not self.env.is_terminal_state(state):
                state, action, reward = self.compute_policy(state, action)
                state_action_reward.append((state, action, reward))

                state = self.env.new_state_given_action(state, action)
                action = self.get_action()
                
            ##########
            ## COmpute return
            ##########
            G = 0
            first = True
            states_actions_returns = []
            for state, action, reward in reversed(state_action_reward):
                # a terminal state has a value of 0 by definition
                # this is the first state we encounter in the reversed list
                # we'll ignore its return (G) since it doesn't correspond to any move
                if first:
                    first = False
                else:
                    states_actions_returns.append((state, action, G))
                G = self.discount_factor * G + reward
            states_actions_returns.reverse()

            ##########
            ## Update state value
            ##########
            # first_time_visit_state = set()
            for state, action, G in states_actions_returns:
                # if state not in first_time_visit_state:
                    # first_time_visit_state.add(state)
                state_value.loc[state_value.states==state, 'times'] += 1
                state_value.loc[state_value.states==state, 'sum_value'] += G

            a= set(x[0]for x in states_actions_returns)
            ic(states_actions_returns)
            ic(a)
            sys.exit()
            # assert len(first_time_visit_state)<=len(states_actions_returns)
        ##########
        ## Update state value
        ##########
        state_value['average'] = state_value.sum_value/state_value.times
        print(state_value)
        for grid_state in state_value.states.unique():
            self.env.grid[grid_state] =state_value[state_value.states == grid_state].average 

        self.env.render_state_value()


def mc_prediction(env, num_episodes, discount_factor=1.0):
    """
    # 1. Monte Carlo Prediction to estimate state-action values

    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """


    def sample_policy(observation):
        """
        A policy that sticks if the player score is >= 20 and hits otherwise.
        """
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1


    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # The final value function
    V = defaultdict(float)     
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for _ in range(100):
            action = sample_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = {tuple(x[0]) for x in episode}
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum(
                x[2] * (discount_factor ** i)
                for i, x in enumerate(episode[first_occurence_idx:])
            )

            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V    


