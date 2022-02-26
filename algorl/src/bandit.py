"""Simple script to run snips of code"""
# Standard Libraries
from pathlib import Path

# Third party libraries
import string
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt

# Local imports
from ..logs import logging
from .tool_box import create_directory
import sys
class Bandits():
    def __init__(
        self, 
        number_of_arms:int = 10,
        q_mean:List[float] = None,
        q_sd:List[float] = None, initial:float=.0,
        bandit_name:List[str]=None, images_dir:str = 'images') -> None:

        self.number_of_arms = number_of_arms
        self.bandit_name = list(string.ascii_uppercase[:self.number_of_arms]) if bandit_name is None else bandit_name

        # real reward for each action
        self.q_mean = np.random.randn(self.number_of_arms) if q_mean is None else q_mean
        self.q_sd = [1] * self.number_of_arms if q_sd is None else q_sd
        self.initial = initial
        self.bandit_df = pd.DataFrame(
            {'mean': self.q_mean, 
            'sd': self.q_sd,
            'action_count': .0,
            'q_estimation': .0 + self.initial,
            },
            index=self.bandit_name).T
        self.images_dir = images_dir
        create_directory(directory_path = self.images_dir)

    def reset_bandit_df(self):
        self.bandit_df.loc['action_count', :]  = .0
        self.bandit_df.loc['q_estimation', :]  = .0 + self.initial

    def plot_bandits (self):
        df = pd.DataFrame(
            {name:np.random.normal(mu, sigma, size=1_000) 
            for name, mu, sigma in zip(self.bandit_name, self.q_mean, self.q_sd)})
        plt.violinplot(dataset=df, showmeans=True)
        plt.xlabel("Action")
        plt.ylabel("Reward distribution")
        plt.xticks(range(1, self.number_of_arms+1), self.bandit_name)
        plt.savefig(Path(self.images_dir, f'{self.number_of_arms}-bandits.png'), dpi=300)
        plt.close()

    def return_bandit_df(self):
        return self.bandit_df


class Greedy():
    """
    This code allows pure, epsilon-greedy with action value or step size with or without optimistic initial values. 
    Page 32 of Sutton and Barto.

    A simple bandit algorithm
    Initialize, for a = 1 to k: 
    Q(a) <- 0
    N(a) <- 0
    Loop for⇢ever:
        A <- either: arg max_a Q(a) with probability 1 - e (breaking ties randomly)
            or: a random action with probability e
        R <- bandit(A)
        N(A) N(A)+1
        Q(A) Q(A)+ 1/N(A) * (R - Q(A))
    """
    def __init__(
        self, bandit:Bandits, epsilon:float=.1, 
        sample_averages:bool=True, step_size:float=0.1
        ) -> None:
        self.bandit = bandit
        self.epsilon = epsilon
        self.sample_averages = sample_averages
        self.step_size = step_size
    
    def _act(self) -> None:
        """
        This function returns the action to be taken based on the epsilon greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandit.bandit_name)
        return np.random.choice(
                self.bandit.bandit_df.loc['q_estimation', :][self.bandit.bandit_df.loc['q_estimation', :] ==\
                    self.bandit.bandit_df.loc['q_estimation', :].max()].index)


    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        reward = np.random.normal(
            self.bandit.bandit_df[action]['mean'], 
            self.bandit.bandit_df[action]['sd'], size=1)[0]
        self.bandit.bandit_df[action]['action_count'] += 1

        if self.sample_averages:
            self.bandit.bandit_df[action]['q_estimation']  =\
                self.bandit.bandit_df[action]['q_estimation']+\
                (reward - self.bandit.bandit_df[action]['q_estimation'])/\
                    self.bandit.bandit_df[action]['action_count']
        else:
            self.bandit.bandit_df[action]['q_estimation'] =\
                self.bandit.bandit_df[action]['q_estimation'] +\
                    self.step_size *\
                         (reward - self.bandit.bandit_df[action]['q_estimation'])

    def simulate(self, time:int)-> None:
        """
        This function simulates the action taking process.
        """
        best_action_count = 0
        best_action_percentage = []
        for num in range(time):
            action = self._act()
            self._step(action)
            if action == self.bandit.bandit_df.idxmax(axis=1)['mean']:
                best_action_count += 1
                best_action_percentage.append(best_action_count/(num+1))
        self.best_action_percentage = best_action_percentage

    def plot_action_taking(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.best_action_percentage)
        plt.xlabel("Steps")
        plt.ylabel("% of time best action is taken")
        plt.savefig(Path(self.bandit.images_dir, f'{self.epsilon}-greedy.png'), dpi=300)
        plt.close()


class UCB():
    """
    Upper Confidence Bound (UCB) algorithm.
    Page 35 of Sutton and Barto.
    """
    def __init__(
        self, bandit:Bandits, 
        sample_averages:bool=True, step_size:float=0.1, UCB_param:float=0.1, epsilon:float=.1
        ) -> None:
        self.bandit = bandit
        self.UCB_param = UCB_param
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.epsilon = epsilon

    def _act(self, num:int) -> None:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandit.bandit_name)

        UCB_estimation = self.bandit.bandit_df.loc['q_estimation', :] + \
            self.UCB_param * np.sqrt(
                np.log(num + 1) / (self.bandit.bandit_df.loc['action_count', :] + 1e-5))
        
        return self.bandit.bandit_df.columns[np.random.choice(np.where(UCB_estimation == np.max(UCB_estimation))[0])]
        

    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        reward = np.random.normal(
            self.bandit.bandit_df[action]['mean'], 
            self.bandit.bandit_df[action]['sd'], size=1)[0]
        self.bandit.bandit_df[action]['action_count'] += 1

        if self.sample_averages:
            self.bandit.bandit_df[action]['q_estimation']  =\
                self.bandit.bandit_df[action]['q_estimation']+\
                (reward - self.bandit.bandit_df[action]['q_estimation'])/\
                    self.bandit.bandit_df[action]['action_count']
        else:
            self.bandit.bandit_df[action]['q_estimation'] =\
                self.bandit.bandit_df[action]['q_estimation'] +\
                    self.step_size *\
                         (reward - self.bandit.bandit_df[action]['q_estimation'])

    def simulate(self, time:int)-> None:
        """
        This function simulates the action taking process.
        """
        best_action_count = 0
        best_action_percentage = []
        for num in range(time):
            action = self._act(num)
            self._step(action)
            if action == self.bandit.bandit_df.idxmax(axis=1)['mean']:
                best_action_count += 1
                best_action_percentage.append(best_action_count/(num+1))
        self.best_action_percentage = best_action_percentage

    def plot_action_taking(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.best_action_percentage)
        plt.xlabel("Steps")
        plt.ylabel("% of time best action is taken")
        plt.savefig(Path(self.bandit.images_dir, 'UCB.png'), dpi=300)
        plt.close()


class GBA(): #TODO
    """
    Gradient Bandit Algorithm
    Page 37 of Sutton and Barto.
    """
    def __init__(
        self, bandit:Bandits, 
        sample_averages:bool=True, step_size:float=0.1, gradient_baseline=True, epsilon:float=.1
        ) -> None:
        self.bandit = bandit
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.gradient_baseline = gradient_baseline
        self.epsilon = epsilon
        self.average_reward = 0

    def _act(self):
        exp_est = np.exp(self.bandit.bandit_df.loc['q_estimation', :])
        self.action_prob = exp_est / np.sum(exp_est)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandit.bandit_name)

        return np.random.choice(self.bandit.bandit_df.columns, p=self.action_prob)

    def _step(self, action, num) -> None:
        """
        This function updates the action value estimates.
        """
        reward = np.random.normal(
            self.bandit.bandit_df[action]['mean'], 
            self.bandit.bandit_df[action]['sd'], size=1)[0]
        self.bandit.bandit_df[action]['action_count'] += 1
        self.average_reward =+ (reward - self.average_reward)/(num+1)
        
        one_hot = np.zeros(self.bandit.bandit_df.shape[1])
        # print()

        one_hot[string.ascii_uppercase.index(action)] = 1

        
        baseline = self.average_reward if self.gradient_baseline else 0

        self.bandit.bandit_df.loc['q_estimation', :] =\
            [(self.bandit.bandit_df[action]['q_estimation']+ self.step_size * (reward - baseline))*x 
            for x in one_hot - self.action_prob]


    def simulate(self, time:int)-> None:
        """
        This function simulates the action taking process.
        """
        best_action_count = 0
        best_action_percentage = []
        for num in range(time):
            action = self._act()
            self._step(action, num)
            if action == self.bandit.bandit_df.idxmax(axis=1)['mean']:
                best_action_count += 1
                best_action_percentage.append(best_action_count/(num+1))
        self.best_action_percentage = best_action_percentage

    def plot_action_taking(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.best_action_percentage)
        plt.xlabel("Steps")
        plt.ylabel("% of time best action is taken")
        plt.savefig(Path(self.bandit.images_dir, 'GBA.png'), dpi=300)
        plt.close()