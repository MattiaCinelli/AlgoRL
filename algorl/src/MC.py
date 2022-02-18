"""Simple script to run snips of code"""
# Standard Libraries
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
    def __init__(
        self, env, step_cost:float = -1, discount_factor:float = 0.9, 
        num_episodes:int = 100, num_of_epochs:int = 10,
        noise:float = .0, plot_name:str = 'grid_world'
        ):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - step_cost: float: cost of moving in the environment
        - gamma: float: discount factor
        - num_episodes: int: number of epochs to run the algorithm
        - num_of_iterations: int: number of iterations within each epoch
        - noise: float: probability of taking a action that is not the one chosen
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
        for _ in range(self.num_of_epochs):
            # compute random first state
            state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]]
            # compute random first action for first state
            action = self.get_action()
            # initialize state value dateframe
            df = pd.DataFrame(columns=['state', 'action', 'reward'])
            G = 0
            for _ in range(self.num_episodes):
                state, action, reward = self.compute_policy(state, action)
                df = df.append({
                    'state' : state,
                    'action' : action,
                    'reward' : reward} , ignore_index=True)

                # new_state = self.env.new_state_given_action(state, action)
                # while self.env.is_terminal_state(new_state):
                #     action = self.get_action()
                #     new_state = self.env.new_state_given_action(state, action)

                new_state = self.env.new_state_given_action(state, action)
                if self.env.is_terminal_state(new_state):
                    break
                state = new_state
                action = self.get_action()

            # ic(df)
            for num in range(df.shape[0]):
                G = df.reward[num] + self.discount_factor*G
                df.loc[num, 'G'] = G
            for grid_state in df.state.unique():
                self.env.grid[grid_state] = df[df.state == grid_state].reward.mean()

            ic(df)
        ic(self.env.grid)
        return df




class MCExploringStarts():
    '''
    Monte Carlo Prediction to estimate state-action values
    '''
    def __init__(
        self, env, discount_factor:float = 0.9, 
        num_episodes:int = 100, num_of_epochs:int = 1,

        ):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - step_cost: float: cost of moving in the environment
        - gamma: float: discount factor
        - num_episodes: int: number of epochs to run the algorithm
        - num_of_iterations: int: number of iterations within each epoch
        - noise: float: probability of taking a action that is not the one chosen
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
        return state, action, -1+self.env.grid[self.env.new_state_given_action(state, action)]

    def compute_state_value(self):
        for _ in range(self.num_of_epochs):
            # compute random first state
            state = self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]]
            # compute random first action for first state
            action = self.get_action()
            # initialize state value dateframe
            df = pd.DataFrame(columns=['state', 'action', 'reward'])
            G = 0
            for _ in range(self.num_episodes):
                state, action, reward = self.compute_policy(state, action)
                df = df.append({
                    'state' : state,
                    'action' : action,
                    'reward' : reward} , ignore_index=True)

                # new_state = self.env.new_state_given_action(state, action)
                # while self.env.is_terminal_state(new_state):
                #     action = self.get_action()
                #     new_state = self.env.new_state_given_action(state, action)

                new_state = self.env.new_state_given_action(state, action)
                if self.env.is_terminal_state(new_state):
                    break
                state = new_state
                action = self.get_action()

            # ic(df)
            for num in range(df.shape[0]):
                G = df.reward[num] + self.discount_factor*G
                df.loc[num, 'G'] = G
            
            for grid_state in df.state.unique():
                self.env.grid[grid_state] = df[df.state == grid_state].reward.mean()

        final = df.groupby(['state', 'action']).G.mean()
        ic(final)
        ic(self.env.grid)
        

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

# V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
# V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
