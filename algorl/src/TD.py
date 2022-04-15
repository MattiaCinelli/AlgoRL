# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import random
import pandas as pd
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

# Local imports
from ..logs import logging
from .tool_box import create_directory, RLFunctions
from matplotlib.table import Table

# 1. Monte Carlo Prediction to estimate state-action values
# 2. On-policy first-visit Monte Carlo Control algorithm
# 3. Off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm

# Source:
# https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf

class TemporalDifferenceFunctions(RLFunctions):
    """
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        RLFunctions.__init__(self)

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

    def sarsa_formula(self, q_values_df, state, action, reward, next_state, next_action):
        '''Q(S, A) <- Q(S, A) + alpha[R + gamma * Q(S', A') - Q(S, A)]'''
        # Q[state][action] = Q[state][action] + alpha * reward + gamma * Q[next_state][next_action] - Q[state][action]
        q_values_df.at[state, action] =\
            q_values_df.at[state, action] + self.alpha *\
                (reward + self.gamma * q_values_df.at[next_state, next_action] \
                    - q_values_df.at[state, action])
        return q_values_df

    def q_learning_formula(self, q_values_df, state, action, reward, next_state, _):
        '''Q(S, A) <- Q(S, A) + alpha[R + gamma * max[Q(S', a)] - Q(S, A)]'''
        # Q[state][action] = Q[state][action] + alpha * reward + gamma * Q[next_state].max() - Q[state][action]
        max_value = max(q_values_df.at[next_state, x] for x in self.env.possible_actions)
        q_values_df.at[state, action] =\
            q_values_df.at[state, action] + self.alpha *\
                (reward+ self.gamma * max_value - q_values_df.at[state, action])
        return q_values_df

    def td_control(self, algo, plot_name:str):
        # Initialize Q(s,a)
        q_values_df = pd.DataFrame(
            0, columns=self.env.possible_actions, index=self.env.all_states, dtype=float)

        # Initialize Q(s,a), for all s element of S+, a element of A(s), arbitrarily except that Q(terminal,·) = 0
        state_action_pairs = {state:self.get_random_action() for state in self.env.all_states}
        # Initialize actions using epsilon greedy and value state of the greed         
        for x, y in state_action_pairs.items():
            state_action_pairs[x] = self.epsilon_greedy(y)

        # Loop for each episode:
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                self.logger.info(f'\tEpoch {epoch}')

            # Get first state and action for the episode
            state = random.choice(self.env.available_states) if self.starting_state is None else self.starting_state
            action = state_action_pairs[state]

            # Loop for each step of the episode:
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                # Generate trajectory
                next_state = self.env.new_state_given_action(state, action)
                next_action = self.epsilon_greedy(state_action_pairs[next_state])
                reward = self.env.grid[next_state]

                # Compute the pair state-actions
                q_values_df = algo(q_values_df, state, action, reward, next_state, next_action)

                self.logger.debug(f'S: {state}, A: {action}, R:{q_values_df.at[state, action]:.3f}, S: {next_state}, A: {next_action}')
                state_action_pairs[next_state] = next_action
                state = next_state
                action = next_action

                num_of_steps += 1

        self.drew_policy(q_values_df, plot_name=plot_name)


class TabularTD0(TemporalDifferenceFunctions):
    '''
    Monte Carlo Prediction to estimate state-action values
    Page 120 of Sutton and Barto.

    Input: the policy \pi to be evaluated
    Algorithm parameter: step size alpha element of (0, 1]
    Initialize V(s), for all s element S+, arbitrarily except that V(terminal) = 0
    Loop for each episode: 
        Initialize S
        Loop for each step of episode:
            A action given by pi for S
            Take action A, observe R, S_{0}
            V(S) <- V(S)+alpha[R+gammaV(S')-V(S)]
        until S is terminal
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9, starting_state=(2,0),
        num_of_epochs:int = 1_00, plot_name='TD0', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        super().__init__()
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state
        self.plot_name = plot_name
        self.reward = reward
        self.logger.info('TD0 initialized')

    def compute_state_value(self):
        self.logger.info('Compute TD0')
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                self.logger.info(f'\tEpoch {epoch}')

            state = self.starting_state
            done = False
            while not done:
                action = self.get_random_action()
                next_state = self.env.new_state_given_action(state, action)
                self.logger.debug(f'state: {state}, action: {action}, new state: {next_state}')
                # V[state] = V[state] + alphas * reward + gamma * V[next_state] * (not done) - V[state]
                self.env.grid[state] = self.env.grid[state] +\
                    self.alpha * (self.reward + self.gamma * self.env.grid[next_state] * (not done) - self.env.grid[state])
                state = next_state
                done = self.env.is_terminal_state(state)


class Sarsa(TemporalDifferenceFunctions):
    '''
    Sarsa (on-policy TD control) for estimating Q=q*
    Page 130 of Sutton and Barto.

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
        self, env, alpha:float = 0.5, gamma:float = 0.9,  starting_state=(2,0), 
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='Sarsa', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        super().__init__()
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state
        self.plot_name = plot_name
        self.epsilon = epsilon
        self.reward = reward
        self.num_episodes = num_episodes
        self.logger.info('SARSA initialized')

    def compute_state_value(self, plot_name='SARSA'):
        self.logger.info('Compute SARSA')
        self.td_control(algo = self.sarsa_formula, plot_name=plot_name)


class QLearning(TemporalDifferenceFunctions):
    '''
    Q-learning for estimating pi=pi*
    Page 131 of Sutton and Barto.

    Algorithm parameters: step size alpha element of (0, 1], small epsilon > 0
    Initialize Q(s,a), for all s element of S+,a element of A(s), arbitrarily except that Q(terminal,·) = 0

    Loop for each episode: 
        Initialize S
        Loop for each step of episode:
            Choose A' from S' using policy derived from Q (e.g., epsilon-greedy) 
            Tale action A, observe R, S'
            Q(S, A) <- Q(S, A) + alpha[R + gamma * max[Q(S', a)] - Q(S, A)]
            S <- S'; A <- A';
    until S is terminal
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9,  starting_state=(2,0), 
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='Qlearning', reward = -1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        super().__init__()
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state
        self.plot_name = plot_name
        self.epsilon = epsilon
        self.reward = reward
        self.num_episodes = num_episodes
        self.logger.info('Q-Learning initialized')

    def compute_state_value(self, plot_name='QLearning'):
        self.logger.info('Compute Q-Learning')
        self.td_control(algo = self.q_learning_formula, plot_name=plot_name)


class NStepTD(TemporalDifferenceFunctions):
    '''
    n-step TD for estimating vi=vi*
    Page 144 of Sutton and Barto.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9, starting_state:tuple=None,
        num_of_epochs:int = 1_00, plot_name='TD0', step_cost = -1, n_step = 1):
        """
        Initializes the grid world
        - env: grid_environment: A tabular environment created by Make class
        - discount_factor: float: discount factor
        - num_of_epochs: int: number of epochs 
        """
        super().__init__()
        self.num_of_epochs = num_of_epochs
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.starting_state = env.initial_state if starting_state is None else starting_state
        self.plot_name = plot_name
        self.step_cost = step_cost
        self.n_step = n_step
        self.logger.info('NStepTD initialized')

    def compute_state_value(self):
        '''Too formulaic, it needs to be refactored''' # TODO
        self.logger.info('Compute NStepTD')
        self.logger.info(f'n: {self.n_step}')
        for epoch in range(self.num_of_epochs):
            if epoch % (self.num_of_epochs/10) == 0:
                self.logger.info(f'\tEpoch {epoch}')

            T = 100_000 # It's the last time step of the episode
            t = 0
            states, rewards = [], []
            state = self.starting_state
            done = False
            while not done:
                if t < T:
                    action = self.get_random_action()
                    self.logger.debug(f'State: {state}, Action: {action}')
                    next_state = self.env.new_state_given_action(state, action)
                    self.logger.debug(f'Next State: {next_state}')
                    reward = self.env.grid[next_state] + self.step_cost 
                    states.append(state)
                    rewards.append(reward)
                    if self.env.is_terminal_state(state):
                        T = t + 1 
                        done = True

                r = t - self.n_step + 1
                if r >= 0:
                    G = sum(rewards[x] * self.gamma**(r-x) for x in range(r, np.min([r+self.n_step, T])))
                    if r + self.n_step < T:
                        G += self.gamma**(self.n_step) * self.env.grid[states[r+self.n_step-1]]

                    if not self.env.is_terminal_state(state):
                        self.env.grid[states[r]] = self.env.grid[states[r]] + self.alpha *\
                            (G - self.env.grid[states[r]])

                if r == (T - 1):
                    done = True
                    self.logger.info(f'\tEpoch {epoch} done (r == (T - 1)')
                t += 1
                state = next_state
        