"""Simple script to run snips of code"""
# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from icecream import ic

import gym
import numpy as np
from .grid_environment import MakeGymGrid

# Local imports
from ..logs import logging

class DP(object):
    '''
    Policy Iteration (using iterative policy evaluation) for estimating pi = pi*
    Page 80 of Sutton and Barto.

    Policy Iteration (using iterative policy evaluation) for estimating pi in pi*
    1. Initialization
        V(s) element of R and pi(s) element of A(s) arbitrarily for all s element of S; V(terminal)= 0
    2. Policy Evaluation 
        Loop:
            Delta <- 0
            Loop for each s element of S:
                v <- V(s)
                V(s) <- sum{s', r}p(s', r| s, pi(s))[r + gamma*V(s')]
                delta <- max(delta, |v delta V (s)|)
        until delta < theta (a small positive number determining the accuracy of estimation)
    3. Policy Improvement
        policy-stable true 
        For each s element of S:
            old-action <- pi(s)
            pi(s) <- underset{a}{argmax} sum_{s', r} p(s', r | s, pi(s))[r + gamma V(s')]
            If old-action != pi(s), then policy-stable false
        If policy-stable, then stop and return V almost equal to v* and pi almost equal to pi*; else go to 2
    '''
    def __init__(
        self, env, step_cost:float = -1, gamma:float = 0.5, 
        noise:float = .0, epsilon:float = 1e-4
        ) -> None:
        """
        Initializes the grid world
        Args:
        -------------------
        - env: grid_environment: A tabular environment created by MakeGrid class
        - step_cost: float: cost of moving in the environment
        - gamma: float: discount factor
        - noise: float: probability of taking a action that is not the one chosen
        - epsilon: float: threshold for convergence
        """
        # Agent position and reward set up
        self.step_cost = step_cost
        self.gamma = gamma
        self.epsilon = epsilon
        self.noise = noise
        self.env = env

        self.logger = logging.getLogger("DynamicsProgramming")
        self.logger.info("Running DP")

    def prob_of_action(self, action:str) -> Dict[str, float]:
        """
        Returns the probability of taking an action
        """
        correct = 1 - self.noise
        wrong = self.noise/2
        return {
            'U': {'U':correct, 'L':wrong, 'R':wrong}, 
            'D': {'D':correct, 'L':wrong, 'R':wrong},
            'L': {'L':correct, 'U':wrong, 'D':wrong},
            'R': {'R':correct, 'U':wrong, 'D':wrong},
            }[action]

    def get_sweep(self) -> Tuple[np.ndarray, bool]:
        '''
        Returns the state value function and whether the policy is stable
        '''
        new_grid = self.env.grid.copy()
        for state in self.env.available_states:
            exploration = []
            
            for action in self.env.possible_actions:
                possible_move = self.prob_of_action(action)

                exploration.append(
                    sum(
                        self.env.grid[self.env.next_state_given_action(state, move)]*possible_move.get(move)
                        for move in possible_move
                        )
                )

            option_results = pd.DataFrame([exploration], columns=self.env.possible_actions)

            new_grid[state] = self.step_cost + self.gamma*\
                option_results[option_results.idxmax(axis=1)[0]][0]

        if np.array_equal(new_grid, self.env.grid):
            return new_grid, True
        elif np.nansum(np.abs(new_grid - self.env.grid)) < self.epsilon :
            return new_grid, True
        else:
            return new_grid, False

    def simulate(self) -> np.ndarray:
        '''
        Simulates the policy iteration algorithm
        '''
        done = False
        while not done:
            self.env.grid, done = self.get_sweep()
        return self.env.grid


class DPv2(object):
    '''
    Policy Iteration (using iterative policy evaluation) for estimating pi = pi*
    Page 80 of Sutton and Barto.
    As for class DP, but with implementation suitable for gym grid environments
    '''
    def __init__(
        self, env:gym.Env, gym_env_name:str, step_cost:float = -1, gamma:float = 0.5, 
        noise:float = .0, epsilon:float = 1e-4
        ) -> None:
        """
        Initializes the grid world
        Args:
        -------------------
        - env: grid_environment: A tabular environment created by MakeGrid class
        - step_cost: float: cost of moving in the environment
        - gamma: float: discount factor
        - noise: float: probability of taking a action that is not the one chosen
        - epsilon: float: threshold for convergence
        """
        self.logger = logging.getLogger("DP Gym env")
        self.logger.info("Running DPv2")
        
        # Agent position and reward set up
        self.gym_env_name = gym_env_name
        self.step_cost = step_cost
        self.gamma = gamma
        self.epsilon = epsilon
        self.noise = noise
        mgg = MakeGymGrid(env)
        self.env = mgg.get_env()
        ic(self.env.desc)
        mgg.drew_tabular_environment()


    # def clear():
    #     _ = system('cls') if name == 'nt' else system('clear') 

    # def act(self, env, gamma, policy, state, v):
    #     for action, action_prob in enumerate(policy[state]):            
    #         for state_prob, next_state, reward, end in env.P[state][action]:                                
    #             v += action_prob * state_prob * (reward + gamma * self[next_state])
    #             self[state] = v
                
    # def evaluate(self, action_values, env, gamma, state):
    #     for action in range(env.nA):
    #         for prob, next_state, reward, terminated in env.P[state][action]:
    #             action_values[action] += prob * (reward + gamma * self[next_state])
    #     return action_values

    # def lookahead(self, env, state, V, gamma):
    #     action_values = np.zeros(env.nA)
    #     return self.evaluate(V, action_values, env, gamma, state)

    # def improve_policy(self, env, gamma=1.0, terms=1e9):    
    #     policy = np.ones([env.nS, env.nA]) / env.nA
    #     evals = 1
    #     for _ in range(int(terms)):
    #         stable = True
    #         V = self.eval_policy(policy, env, gamma=gamma)
    #         for state in range(env.nS):
    #             current_action = np.argmax(policy[state])
    #             action_value = self.lookahead(env, state, V, gamma)
    #             best_action = np.argmax(action_value)
    #             if current_action != best_action:
    #                 stable = False                
    #                 policy[state] = np.eye(env.nA)[best_action]
    #             evals += 1                
    #             if stable:
    #                 return policy, V

    # def eval_policy(self, policy, env, gamma=1.0, theta=1e-5, terms=1e9):     
    #     V = np.zeros(env.nS)
    #     delta = 0
    #     for _ in range(int(terms)):
    #         for state in range(env.nS):            
    #             self.act(V, env, gamma, policy, state, v=0.0)
    #         self.clear()
    #         print(V)
    #         time.sleep(1)
    #         v = np.sum(V)
    #         if v - delta < theta:
    #             return V
    #         else:
    #             delta = v
    #     return V

    # def value_iteration(self, env, gamma=1.0, theta=1e-9, terms=1e9):
    #     V = np.zeros(env.nS)
    #     for _ in range(int(terms)):
    #         delta = 0
    #         for state in range(env.nS):
    #             action_value = self.lookahead(env, state, V, gamma)
    #             best_action_value = np.max(action_value)
    #             delta = max(delta, np.abs(V[state] - best_action_value))
    #             V[state] = best_action_value
    #         if delta < theta: break
    #     policy = np.zeros([env.nS, env.nA])
    #     for state in range(env.nS):
    #         action_value = self.lookahead(env, state, V, gamma)
    #         best_action = np.argmax(action_value)
    #         policy[state, best_action] = 1.0
    #     return policy, V

    # def play(self, env, episodes, policy):
    #     wins = 0
    #     total_reward = 0
    #     for _ in range(episodes):
    #         term = False
    #         state = env.reset()
    #         while not term:
    #             action = np.argmax(policy[state])
    #             next_state, reward, term, info = env.step(action)
    #             total_reward += reward
    #             state = next_state
    #             if term and reward == 1.0:
    #                 wins += 1
    #     average_reward = total_reward / episodes
    #     return wins, total_reward, average_reward
