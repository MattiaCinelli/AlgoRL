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

# 1. Monte Carlo Prediction to estimate state-action values
# 2. On-policy first-visit Monte Carlo Control algorithm
# 3. Off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm

# Source:
# https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf

class TemporalDifferenceFunctions(object):
    """
    """
    def __init__(self) -> None:
        pass
    def get_action(self):
        return np.random.choice(self.env.possible_actions) 

    def compute_policy(self, state, action):
        return state, action, self.env.grid[self.env.new_state_given_action(state, action)]

    def state_action_reward_path(self, state, action):
        state_action_reward = []
        num_of_steps = 0
        while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
            reward = self.env.grid[self.env.new_state_given_action(state, action)]
            state_action_reward.append((state, action, reward))

            state = self.env.new_state_given_action(state, action)
            action = self.get_action()
            num_of_steps += 1
        return state_action_reward
    
    def epsilon_greedy(self, action):
        return np.random.choice(self.env.possible_actions) if np.random.rand() < self.epsilon else action

    def _drew_grid(self, tb, width, height, ax):
        for i in range(self.env.grid_row):
            tb.add_cell(i,-1, width, height, text=i, loc='right', edgecolor='none', facecolor='none',)

        for i in range(self.env.grid_col):
            tb.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none',)
        ax.add_table(tb)

    def drew_policy(self, df, plot_name:str):
        df = df.idxmax(axis=1)
        print(df)
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

                arrows = "${}$".format(arrow_symbols[df[i, j]])
                tb.add_cell(i, j, width, height, text=arrows, loc='center', facecolor='white')

        self._drew_grid(tb, width, height, ax)
        plt.savefig(Path(self.env.images_dir, f'{plot_name}_state_values.png'), dpi=300)


class TabuladTD0(TemporalDifferenceFunctions): # Page 120
    '''
    Monte Carlo Prediction to estimate state-action values

    Input: the policy \pi to be evaluated
    Algorithm parameter: step size alfa element of (0, 1]
    Initialize V(s), for all s element S+, arbitrarily except that V(terminal) = 0
    Loop for each episode: 
        Initialize S
        Loop for each step of episode:
            A action given by pi for S
            Take action A, observe R, S_{0}
            V(S) <- V(S)+alpha[R+gammaV(S')-V(S)]
        until S is terminal
    '''
    def __init__(self, env, alfa:float = 0.5, gamma:float = 0.9,  num_of_epochs:int = 1_00, plot_name='TD0', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alfa = alfa
        self.env = env
        self.plot_name = plot_name
        self.reward = reward

    def compute_state_value(self):
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')
            state = (2,0) # self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            
            while not self.env.is_terminal_state(state):
                # print(state)
                action = self.get_action()
                new_state = self.env.new_state_given_action(state, action)
                
                self.env.grid[state] = self.env.grid[state] + self.alfa * (self.reward + self.gamma * self.env.grid[new_state] - self.env.grid[state])
                state = new_state

        self.env.render_state_value()
        self.env.drew_policy(plot_name=self.plot_name)
        self.env.draw_state_value(plot_name=self.plot_name)


class Sarsa(TemporalDifferenceFunctions): # page 130
    '''
    Sarsa (on-policy TD control) for estimating Q=q* page 130

    Algorithm parameters: step size alpha element of (0, 1], small " > 0
    Initialize Q(s,a), for all s element of S+,a element of A(s), arbitrarily except that Q(terminal,·) = 0
    Loop for each episode: 
        Initialize S
        Choose A from S using policy derived from Q (e.g., "-greedy) 
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., "-greedy) 
            Q(S, A) <- Q(S, A) + alpha[R + gamma*Q(S', A') - Q(S, A)]
            S <- S'; A <- A';
    until S is terminal
    '''
    def __init__(
        self, env, alfa:float = 0.5, gamma:float = 0.9,  
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='Sarsa', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alfa = alfa
        self.env = env
        self.plot_name = plot_name
        self.epsilon = epsilon
        self.reward = reward
        self.num_episodes = num_episodes

    def compute_state_value(self):
        # Initialize Q(s,a
        q_values_df = pd.DataFrame(
            0, columns=self.env.possible_actions, index=self.env.all_states, dtype=float)

        # Initialize actions using epsilon greedy and value state of the greed 
        state_action_pairs = {state:self.get_action() for state in self.env.all_states}
        for x, y in state_action_pairs.items():
            state_action_pairs[x] = self.epsilon_greedy(y)
        # state_action_pairs = {(0, 0): 'R', (0, 1): 'D', (0, 2): 'L', (1, 0): 'U', (1, 2): 'D', (2, 0): 'R', (2, 1): 'R', (2, 2): 'U', (2, 3): 'U'}

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')
            state = (2,0) # self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            action = state_action_pairs[state]
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                new_state = self.env.new_state_given_action(state, action)
                # if self.env.is_terminal_state(new_state):
                #     new_action = action
                # else:
                new_action = self.epsilon_greedy(state_action_pairs[new_state])

                q_values_df.at[state, action] =\
                    q_values_df.at[state, action] + self.alfa *\
                        (self.env.grid[new_state]+ self.gamma * q_values_df.at[new_state, new_action] - q_values_df.at[state, action])
                
                state_action_pairs[new_state] = new_action
                state = new_state
                action = new_action

                num_of_steps += 1
        # ic(q_values_df)
        # ic()
        self.drew_policy(q_values_df, plot_name=self.plot_name)


class QLearning(TemporalDifferenceFunctions): # page 131
    '''
    Q-learning for estimating pi=pi* page 131
    Algorithm parameters: step size alpha element of (0, 1], small " > 0
    Initialize Q(s,a), for all s element of S+,a element of A(s), arbitrarily except that Q(terminal,·) = 0

    Loop for each episode: 
        Initialize S
        Loop for each step of episode:
            Choose A' from S' using policy derived from Q (e.g., "-greedy) 
            Tale action A, observe R, S'
            Q(S, A) <- Q(S, A) + alpha[R + gamma * max[Q(S', a)] - Q(S, A)]
            S <- S'; A <- A';
    until S is terminal
    '''
    def __init__(
        self, env, alfa:float = 0.5, gamma:float = 0.9,  
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='Qlearning', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alfa = alfa
        self.env = env
        self.plot_name = plot_name
        self.epsilon = epsilon
        self.reward = reward
        self.num_episodes = num_episodes

    def compute_state_value(self):
        # Initialize Q(s,a
        q_values_df = pd.DataFrame(
            0, columns=self.env.possible_actions, index=self.env.all_states, dtype=float)

        # Initialize actions using epsilon greedy and value state of the greed 
        state_action_pairs = {state:self.get_action() for state in self.env.all_states}
        for x, y in state_action_pairs.items():
            state_action_pairs[x] = self.epsilon_greedy(y)
        # state_action_pairs = {(0, 0): 'R', (0, 1): 'D', (0, 2): 'L', (1, 0): 'U', (1, 2): 'D', (2, 0): 'R', (2, 1): 'R', (2, 2): 'U', (2, 3): 'U'}

        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                print(f'Epoch {epoch}')
            state = (2,0) # self.env.available_states[np.random.choice(len(self.env.available_states),1 )[0]] 
            action = state_action_pairs[state]
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                new_state = self.env.new_state_given_action(state, action)
                # if self.env.is_terminal_state(new_state):
                #     new_action = action
                # else:
                new_action = self.epsilon_greedy(state_action_pairs[new_state])
                max_value = max(q_values_df.at[new_state, x] for x in self.env.possible_actions)

                q_values_df.at[state, action] =\
                    q_values_df.at[state, action] + self.alfa *\
                        (self.env.grid[new_state]+ self.gamma * max_value - q_values_df.at[state, action])
                
                state_action_pairs[new_state] = new_action
                state = new_state
                action = new_action

                num_of_steps += 1
        # ic(q_values_df)
        # ic()
        self.drew_policy(q_values_df, plot_name=self.plot_name)

class NstepTD(TemporalDifferenceFunctions): # page 144
    '''
    n-step TD for estimating vi=vi* page 144
    '''
    pass