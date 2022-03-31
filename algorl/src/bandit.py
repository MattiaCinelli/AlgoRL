"""Simple script to run snips of code"""
# Standard Libraries
from pathlib import Path

# Third party libraries
import string
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from pyrsistent import v
from icecream import ic

# Local imports
from ..logs import logging
from .tool_box import create_directory
import sys
logger = logging.getLogger(__name__)


class TestAll(object):
    logger.info("Running TestAll MAB")
    def __init__(self, time_steps:int=100, arms:int=5, number_of_trials:int=5, images_dir:str='images', q_mean:List[float] = None, q_sd:List[float] = None):
        self.df_return = pd.DataFrame()
        self.df_action = pd.DataFrame()
        self.time_steps = time_steps
        self.number_of_trials = number_of_trials
        self.arms = arms
        self.images_dir = images_dir
        self.q_mean = np.random.randn(self.arms) if q_mean is None else q_mean
        self.q_sd = [1] * self.arms if q_sd is None else q_sd
    
    def test_algo(self, algo, col_name:str=None):
        if col_name is None:
            col_name = algo.__name__
        logger.info(f"Running {col_name}")
        rewards, best_actions = [], []
        for _ in range(self.number_of_trials):
            logger.info("\ttest: {}".format(_))
            #1 New bandits
            bandit = Bandits(number_of_arms = self.arms)
            #2 Simulate
            explore = algo(bandit)
            reward, best_action =  explore.simulate(time = self.time_steps)
            rewards.append(np.cumsum(reward))
            best_actions.append(best_action)
            bandit.reset_bandit_df()
        self.df_return[f"{col_name}"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(self.time_steps))
        self.df_action[f"{col_name}"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(self.time_steps))
    
    def return_dfs(self):
        return self.df_return, self.df_action

    def _comparing_plots(self, arg0, arg1, pic_name):
        plt.figure(figsize=(10, 5))
        arg0.plot.line()
        plt.xlabel("Steps")
        plt.ylabel(arg1)
        plt.savefig(Path(self.images_dir, f'{pic_name}.png'), dpi=300)
        plt.close()

    def plot_action_taken(self, best_actions, pic_name:str = 'BestActions' ):
        self._comparing_plots(
            best_actions, "% of time best action is taken", pic_name
        )

    def plot_returns(self, tot_return, pic_name:str='TotalReturns' ):
        self._comparing_plots(tot_return, "Total returns", pic_name)


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
            {'target': self.q_mean,
            'true_sd': self.q_sd, # real sd for each action
            'action_count': .0, # number of times action was taken
            'q_estimation': .0 + self.initial, # Mean of rewards after each action
            'estimated_sd': self.q_sd, # Standard deviation of rewards after each action
            },
            index=self.bandit_name).T
        self.images_dir = images_dir
        create_directory(directory_path = self.images_dir)

    def reset_bandit_df(self):
        self.bandit_df.loc['action_count', :] = .0
        self.bandit_df.loc['q_estimation', :] = .0 + self.initial

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

    def plot_true_mean_vs_estimation(self, pic_name:str='TargetVsEstimation', y_axis='q_estimation'):
        '''
        Scatter plot of true mean vs estimation
        '''
        _, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.q_mean, self.bandit_df.loc[y_axis, :], s=100, c=range(len(self.q_mean)))
        ax.set_xlabel("Target")
        ax.set_ylabel("Estimation")
        ax.set_title("Target vs estimated values")
        # Add text labels
        for i, txt in enumerate(self.bandit_name):
            ax.annotate(txt, (self.q_mean[i], self.bandit_df.loc[y_axis, :][i]))
        # Add diagonal line
        ax.plot([min(self.q_mean), max(self.q_mean)], [min(self.q_mean), max(self.q_mean)], 'k--')
        plt.savefig(Path(self.images_dir, f'{pic_name}.png'), dpi=300)
        plt.close()


class BernoulliBandits(Bandits):
    def __init__(
        self, number_of_arms: int = 10, q_mean: List[float] = None, q_sd: List[float] = None, 
        initial: float = 1, bandit_name: List[str] = None, images_dir: str = 'images') -> None:
        ''''''
        q_mean = np.linspace(0.1, 0.9, num=number_of_arms) if q_mean is None else q_mean
        super().__init__(number_of_arms, q_mean, q_sd, initial, bandit_name, images_dir)
        self.bandit_df.index = [
            'target', # Our posterior
            'theta_hat', # Updated posterior
            'action_count', 
            'alpha', # Sucesses
            'beta' # Failures
            ]


class MABFunctions(object):
    def __init__(self) -> None:
        pass

    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        reward = np.random.normal(
            self.bandit.bandit_df[action]['target'], 
            self.bandit.bandit_df[action]['estimated_sd'], size=1)[0]
        self.bandit.bandit_df[action]['action_count'] += 1

        if self.sample_averages:
            self.bandit.bandit_df[action]['q_estimation'] =\
                self.bandit.bandit_df[action]['q_estimation']+\
                (reward - self.bandit.bandit_df[action]['q_estimation'])/\
                    self.bandit.bandit_df[action]['action_count']
        else:
            self.bandit.bandit_df[action]['q_estimation'] =\
                self.bandit.bandit_df[action]['q_estimation'] +\
                    self.step_size *\
                         (reward - self.bandit.bandit_df[action]['q_estimation'])
        return reward

    def simulate(self, time:int)-> None:
        """
        This function simulates the action taking process.
        """
        best_action = self.bandit.return_bandit_df().loc['target', :].idxmax()
        best_action_count = 0
        best_action_percentage = []
        for num in range(time):
            action = self._act(num)
            self.tot_return.append(self._step(action))
            if action == best_action:
                best_action_count += 1
            best_action_percentage.append(best_action_count/(num+1))
        self.best_action_percentage = best_action_percentage
        return self.tot_return, self.best_action_percentage


class OnlyExploration(MABFunctions):
    def __init__(
        self, bandit:Bandits, sample_averages:bool=True, 
        step_size:float=0.1) -> None:
        self.bandit = bandit
        self.tot_return = []
        self.sample_averages = sample_averages
        self.step_size = step_size

    def _act(self, _:int) -> str:
        """
        This function returns a random action 
        """
        return np.random.choice(self.bandit.bandit_name)


class OnlyExploitation(MABFunctions):
    def __init__(
        self, bandit:Bandits, sample_averages:bool=True, 
        step_size:float=0.1) -> None:
        self.bandit = bandit
        self.tot_return = []
        self.sample_averages = sample_averages
        self.step_size = step_size

    def _act(self, _:int) -> str:
        """
        This function returns a random action 
        """
        return self.bandit.return_bandit_df().loc['target', :].idxmax()


class Greedy(MABFunctions):
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
        self.tot_return = []

    def _act(self, _:int) -> str:
        """
        This function returns the action to be taken based on the epsilon greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandit.bandit_name)
        return np.random.choice(
                self.bandit.bandit_df.loc['q_estimation', :][self.bandit.bandit_df.loc['q_estimation', :] ==\
                    self.bandit.bandit_df.loc['q_estimation', :].max()].index)


class UCB(MABFunctions):
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
        self.tot_return = []

    def _act(self, num:int) -> pd.DataFrame:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandit.bandit_name)

        UCB_estimation = self.bandit.bandit_df.loc['q_estimation', :] + \
            self.UCB_param * np.sqrt(
                np.log(num + 1) / (self.bandit.bandit_df.loc['action_count', :] + 1e-5))
        
        return self.bandit.bandit_df.columns[np.random.choice(np.where(UCB_estimation == np.max(UCB_estimation))[0])]


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
            self.bandit.bandit_df[action]['target'], 
            self.bandit.bandit_df[action]['estimated_sd'], size=1)[0]
        self.bandit.bandit_df[action]['action_count'] += 1
        self.average_reward =+ (reward - self.average_reward)/(num+1)
        
        one_hot = np.zeros(self.bandit.bandit_df.shape[1])
        # print()

        one_hot[string.ascii_uppercase.index(action)] = 1

        
        baseline = self.average_reward if self.gradient_baseline else 0

        self.bandit.bandit_df.loc['q_estimation', :] =\
            [(self.bandit.bandit_df[action]['q_estimation']+ self.step_size * (reward - baseline))*x 
            for x in one_hot - self.action_prob]


class ThompsonSampling(MABFunctions):
    """
    The Thompson sampling strategy is a sample- based probability matching strategy that allows us to use 
    Bayesian techniques to balance the exploration and exploitation trade-off.
    A simple way to implement this strategy is to keep track of each Q-value as a Gaussian distribution. Page 122, Grokking.
    P( 𝛍 | data ) = P(data | 𝛍) P(𝛍) / P(data)
              ∝ P(data | 𝛍) P(𝛍)
    posterior ∝ likelihood x prior

    If
        Prior is 𝛍 ~ Beta(𝛼, 𝛽)
        Likelihood is k | 𝛍 ~ Binomial(N, 𝛍)
    Then
        Posterior is 𝛍 ~ Beta(𝛼 + successes, 𝛽 + failures)
    Where
    N = successes + failures
  
    BernTS(K,α,β)
    -----------------
    for t = 1,2,... do 
        #sample model:
        for k = 1,...,K do
            Sample θk ∼ beta(αk,βk)
        end for
    
        #select and apply action:
        xt ← argmaxk θk
        Apply xt and observe rt
        #update distribution:
        (αxt,βxt) ← (αxt + rt,βxt + 1 − rt)
    end for

    BernGreedy
    -----------
    #estimate model: 
    # for k = 1,...,K do
        θk ←αk/(αk +βk)
    end for

    """
    def __init__(
        self, bandit:Bandits, alpha = 1, beta = 1, bandit_type:str='Gaussian',
        epsilon:float=.1, sample_averages:bool=True, step_size:float=0.1
        ) -> None:
        self.bandit = bandit
        self.alpha = alpha
        self.beta = beta
        self.bandit_type = bandit_type
        self.epsilon = epsilon
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.tot_return = []

    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        logger.debug(action)
        # Compute Bernoulli distribution
        reward = np.random.binomial(1, self.bandit.bandit_df[action]['target'], size=1)[0]
        logger.debug(reward)
        self.bandit.bandit_df[action]['action_count'] += 1
        self.bandit.bandit_df[action]['alpha'] += reward
        self.bandit.bandit_df[action]['beta'] += 1-reward
        return reward

    def _act(self, num:int) -> str:        
        if self.bandit_type == 'BernTS':
            # Compute Bernoulli distributions
            self.bandit.bandit_df.loc['theta_hat', :] = \
                np.random.beta(a=self.bandit.bandit_df.loc['alpha', :], b=self.bandit.bandit_df.loc['beta', :])
            logger.debug(self.bandit.bandit_df.loc['theta_hat', :])
        elif self.bandit_type == 'BernGreedy':
            # Compute Bernoulli distributions
            self.bandit.bandit_df.loc['theta_hat', :] = \
                self.bandit.bandit_df.loc['alpha', :]/(self.bandit.bandit_df.loc['alpha', :]+self.bandit.bandit_df.loc['beta', :])
            logger.debug(self.bandit.bandit_df.loc['theta_hat', :])
        else:
            raise ValueError(f'Bandit type {self.bandit_type} not supported')

        # select action
        return np.random.choice(
            self.bandit.bandit_df.loc['theta_hat', :][self.bandit.bandit_df.loc['theta_hat', :] ==\
                self.bandit.bandit_df.loc['theta_hat', :].max()].index)
