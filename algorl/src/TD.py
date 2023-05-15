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

    def select_action(self, state, Q):
        '''
        Selects an action given a state
        '''
        if np.random.rand() < self.epsilon:
            return self.get_random_action(state)
        else:
            return self.env.possible_actions[np.argmax(Q.loc[state, :])]

    def get_q_df(self):
        # Initialize Q(s,a)
        return pd.DataFrame( 0, columns=self.env.possible_actions, index=pd.MultiIndex.from_tuples(self.env.all_states), dtype=float)

    def td_control(self, algo, plot_name:str, lambda_target=False):
        # Initialize Q(s,a)
        q_values_df = self.get_q_df()

        if lambda_target == True:
            eligibility_df = self.get_q_df()

        # Initialize Q(s,a), for all s element of S+, a element of A(s), arbitrarily except that Q(terminal,·) = 0
        state_action_pairs = {state:self.get_random_action(state) for state in self.env.all_states}
        # Initialize actions using epsilon greedy and value state of the greed         
        for x, y in state_action_pairs.items():
            state_action_pairs[x] = self.epsilon_greedy(y)

        # Loop for each episode:
        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'\tEpoch {epoch}')

            # Get first state and action for the episode
            state = random.choice(self.env.available_states) if self.starting_state is None else self.starting_state
            action = state_action_pairs[state]

            # Loop for each step of the episode:
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                # Generate trajectory
                next_state = self.env.next_state_given_action(state, action)
                next_action = self.epsilon_greedy(state_action_pairs[next_state])
                reward = self.env.grid[next_state]

                # Compute the pair state-actions
                if lambda_target:
                    q_values_df, eligibility_df = algo(q_values_df, eligibility_df, state, action, reward, next_state, next_action)
                else:
                    q_values_df = algo(q_values_df, state, action, reward, next_state, next_action)

                self.logger.debug(f'S: {state}, A: {action}, R:{q_values_df.at[state, action]:.3f}, S: {next_state}, A: {next_action}')
                state_action_pairs[next_state] = next_action
                state = next_state
                action = next_action

                num_of_steps += 1


        self.logger.info(q_values_df)

        self.drew_policy(q_values_df, plot_name=plot_name)


class TabularTD0(TemporalDifferenceFunctions):
    '''
    Monte Carlo Prediction to estimate state-action values
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 120.

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
                action = self.get_random_action(state)
                next_state = self.env.next_state_given_action(state, action)
                self.logger.debug(f'state: {state}, action: {action}, new state: {next_state}')
                # V[state] = V[state] + alphas * reward + gamma * V[next_state] * (not done) - V[state]
                self.env.grid[state] = self.env.grid[state] +\
                    self.alpha * (self.reward + self.gamma * self.env.grid[next_state] * (not done) - self.env.grid[state])
                state = next_state
                done = self.env.is_terminal_state(state)


class Sarsa(TemporalDifferenceFunctions):
    '''
    Sarsa (on-policy TD control) for estimating Q=q*
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 130

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
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 131.

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
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 144.
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
                    action = self.get_random_action(state)
                    self.logger.debug(f'State: {state}, Action: {action}')
                    next_state = self.env.next_state_given_action(state, action)
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


class DoubleQLearning(TemporalDifferenceFunctions):
    '''
    Double Q-learning for estimating pi=pi*
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 136.
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 193.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9,  starting_state=(2,0), 
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='D-QL', reward = -1):
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
        self.logger.info('Double Q-Learning initialized')


    def compute_state_value(self, plot_name='DoubleQLearning'):
        self.logger.info('Compute Double Q-Learning')
        # self.td_control(algo = self.q_learning_formula, plot_name=plot_name)
        # Initialize Q(s,a)
        Q1 = self.get_q_df()
        Q2 = self.get_q_df()

        # Loop for each episode:
        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'\tEpoch {epoch}')

            # Get first state and action for the episode
            state = random.choice(self.env.available_states) if self.starting_state is None else self.starting_state
            action = self.select_action(state, (Q1 + Q2)/2)

            # Loop for each step of the episode:
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                # Generate trajectory
                next_state = self.env.next_state_given_action(state, action)
                next_action = self.select_action(next_state, (Q1 + Q2)/2)
                reward = self.env.grid[next_state]

                # Compute the pair state-actions
                if np.random.randint(2):
                    best_action = self.env.possible_actions[np.argmax(Q1.loc[state, :])]
                    td_target = reward + self.gamma * Q2.at[next_state, best_action]
                    td_error = td_target - Q1.at[state, action]
                    Q1.at[state, action] = Q1.at[state, action] + self.alpha * td_error
                    ''' From B&S
                    best_action = self.env.possible_actions[np.argmax(Q1.loc[next_state, :])]
                    Q1.at[state, action] =\
                        Q1.at[state, action] + self.alpha *\
                            (reward + self.gamma * Q2.at[next_state, best_action] - Q1.at[state, action])
                    '''
                else:
                    best_action = self.env.possible_actions[np.argmax(Q2.loc[state, :])]
                    td_target = reward + self.gamma * Q1.at[next_state, best_action]
                    td_error = td_target - Q2.at[state, action]
                    Q2.at[state, action] = Q2.at[state, action] + self.alpha * td_error
                    ''' # From B&S
                    best_action = self.env.possible_actions[np.argmax(Q2.loc[next_state, :])]
                    Q2.at[state, action] =\
                        Q2.at[state, action] + self.alpha *\
                            (reward + self.gamma * Q1.at[next_state, best_action] - Q2.at[state, action])
                    '''

                state = next_state
                action = next_action

                num_of_steps += 1
            self.logger.debug(f'num of steps: {num_of_steps}')
        # After all epochs are done, plot the results
        self.drew_policy((Q1 + Q2), plot_name=plot_name)


class TabularDynaQ(TemporalDifferenceFunctions): #TODO
    '''
    Tabular Dyna-Q
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 164.
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 218.
    '''
    def __init__(
        self, env, alpha:float = 0.5, gamma:float = 0.9,  starting_state=(2,0), 
        num_of_epochs:int = 1_000, num_episodes =10_000, epsilon = 0.1,
        plot_name='TabularDynaQ', reward = -1):
        """
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
        self.logger.info('Tabular Dyna-Q initialized')

    def compute_state_value(self, plot_name='TabularDynaQ'):
        self.logger.info('Compute Tabular Dyna-Q')
        # self.td_control(algo = self.q_learning_formula, plot_name=plot_name)
        # Initialize Q(s,a)
        Q = self.get_q_df()

        # Loop for each episode:
        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'\tEpoch {epoch}')

            # Get first state and action for the episode
            state = random.choice(self.env.available_states) if self.starting_state is None else self.starting_state
            action = self.select_action(state, Q)

            # Loop for each step of the episode:
            num_of_steps = 0
            while not self.env.is_terminal_state(state) and num_of_steps < self.num_episodes:
                # Generate trajectory
                next_state = self.env.next_state_given_action(state, action)
                next_action = self.select_action(next_state, Q)
                reward = self.env.grid[next_state]


class BackwardsViewTDLambda(RLFunctions):
    """
    Backwards View TD(Lambda) Algorithm for computing the state-value function of a given policy.
    Policy is a random policy as default. 
    Reference:
    --------------------
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 158.
    """

    def __init__(
        self, env, alpha:float = 0.3, gamma:float = 0.9, lambda_:float = 0.2, 
        starting_state:tuple = (2,0), num_of_epochs:int = 1_000, num_episodes:int = 10_000,
        reward:int = -1, plot_name='BV-TD(Lambda)'):
        """
        """
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.starting_state = env.initial_state if starting_state == None else starting_state
        self.num_of_epochs = num_of_epochs
        self.num_of_episodes = num_episodes
        self.plot_name = plot_name
        self.reward = reward

    def update_state_values(self, state_values:pd.DataFrame, state:tuple, state_prime:tuple, reward:float):
        # Compute the TD target.
        td_target = reward + self.gamma * state_values.loc[state_values.states == state_prime, 'state_value'].to_numpy() # Changing to np avoids 0s being turned into NaNs.
        # Compute the TD error.  
        td_error = td_target - (state_values.loc[state_values.states == state, 'state_value']).to_numpy()
        # Update the states.
        updated_values = state_values.loc[:, 'eligibility'].to_numpy() * self.alpha * td_error
        state_values.loc[:, 'state_value']+= updated_values

        return state_values


    def compute_state_values(self):

        state_values = pd.DataFrame({"states":self.env.all_states, 'times':0, 'eligibility':0, 'state_value':0})

        for epoch in range(self.num_of_epochs):
            if epoch % 100 == 0:
                self.logger.info(f'Epoch {epoch}')

            # Initialise the starting state and reset the eligibility data frame.
            state = self.starting_state
            state_values.loc[:, 'eligibility'] = 0

            # Until we hit a terminal state. 
            while not self.env.is_terminal_state(state):

                # Compute the action, state prime, reward and increment the eligibility vector. 
                action = self.get_random_action(state)
                state_prime = self.env.next_state_given_action(state, action)
                reward = self.env.grid[self.env.next_state_given_action(state, action)] - 1
                state_values.loc[state_values.states == state, ['times', 'eligibility']] += 1

                # Call the compute state function. 
                state_values = self.update_state_values(state_values, state, state_prime, reward)

                # Decay the eligibility.
                state_values.loc[:, 'eligibility'] = state_values.loc[:, 'eligibility'] * (self.lambda_ * self.gamma)

                # Update the state. 
                state = state_prime

        # For plotting. 
        state_values.drop(state_values[state_values.times == 0].index, inplace=True)
        state_values.reset_index(drop=True, inplace=True)
        for grid_state in state_values.states.unique():
            self.env.grid[grid_state]=state_values[state_values.states == grid_state].state_value
                

class SarsaLambda(TemporalDifferenceFunctions):
    """ 
    SARSA Algorithm that utilizes the lambda target instead of the TD target.
    Trace vector can be set as either accumulating or replacing. 
    Policy is epsilon-greedy. 
    Reference:
    --------------------
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 210.
    """

    def __init__(
            self, env, alpha:float = 0.5, gamma:float = 0.9, lambda_:float = 0.5, 
            epsilon:float = 0.1, replacing_traces:bool = False, starting_state:tuple = (2,0), 
            num_of_epochs:int = 1_000, num_episodes:int = 10_000, reward:int = -1,
            plot_name:str = 'Sarsa_Lambda'):
        """
        """
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.replacing_traces = replacing_traces
        self.starting_state = env.initial_state if starting_state == None else starting_state
        self.num_of_epochs = num_of_epochs
        self.num_episodes = num_episodes
        self.plot_name = plot_name
        self.reward = reward
        self.logger.info('Sarsa Lambda initialized')

            
    def sarsa_lambda_formula(self, q_values_df, eligibility_df, state, action, reward, next_state, next_action):

        if self.replacing_traces == True:
            eligibility_df.at[state, action] += 1
            eligibility_df.clip(0.0, 1.0, inplace=True)
        else:
            eligibility_df.at[state, action] += 1 # Implement accumulating trace. 

        # Update the Q values. 
        td_error = reward + self.gamma * q_values_df.at[next_state, next_action] - q_values_df.at[state, action]
        q_values_df = q_values_df + self.alpha * td_error * eligibility_df # Update all the state-action pairs.

        # Decay the eligibility values.
        eligibility_df = (self.gamma * self.lambda_) * eligibility_df

        return q_values_df, eligibility_df

    def compute_state_value(self, plot_name='SARSA-Lambda'):
        self.logger.info('Compute SARSA Lambda')
        self.td_control(algo=self.sarsa_lambda_formula, plot_name=plot_name, lambda_target=True)

    

class WatkinsQ(TemporalDifferenceFunctions): # AKA Q Learning Lambda.
    """ 
    Q Learning algorithm that utilizes the lambda target instead of the TD target.
    Policy is epsilon-greedy. 
    Reference:
    --------------------
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 214.     
    """

    def __init__(
            self, env, alpha:float = 0.5, gamma:float = 0.9, lambda_:float = 0.5, 
            epsilon:float = 0.1, replacing_traces:bool = False, starting_state:tuple = (2,0), 
            num_of_epochs:int = 1_000, num_episodes:int = 10_000, reward:int = -1,
            plot_name:str = 'Watkins Q', ):
        """
        """
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.replacing_traces = replacing_traces
        self.starting_state = env.initial_state if starting_state == None else starting_state
        self.num_of_epochs = num_of_epochs
        self.num_episodes = num_episodes
        self.plot_name = plot_name
        self.reward = reward
        self.logger.info('Watkins Q initialized')

    def watkins_Q_formula(self, q_values_df, eligibility_df, state, action, reward, next_state, next_action):
        # Increment the eligibility trace. 
        if self.replacing_traces == True:
            eligibility_df.at[state, action] += 1
            eligibility_df.clip(0.0, 1.0, inplace=True)
        else:
            eligibility_df.at[state, action] += 1 # Implement accumulating trace. 

        # Update the Q values.
        max_value = max(q_values_df.at[next_state, x] for x in self.env.possible_actions)
        td_error = reward + self.gamma * max_value - q_values_df.at[state, action]
        q_values_df = q_values_df + self.alpha * td_error * eligibility_df
        
        # Decay the eligibility matrix. 
        eligibility_df = (self.gamma * self.lambda_) * eligibility_df

        return q_values_df, eligibility_df

    def compute_state_value(self, plot_name='SARSA-Lambda'):
        self.logger.info('Compute Watkins Q')
        self.td_control(algo=self.watkins_Q_formula, plot_name=plot_name, lambda_target=True)



