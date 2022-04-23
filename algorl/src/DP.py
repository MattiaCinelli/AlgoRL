"""Simple script to run snips of code"""
# Standard Libraries
import sys
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set

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
