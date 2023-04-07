"""Simple script to run snips of code"""
# Standard Libraries
from pathlib import Path

# Third party libraries
import string
import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from icecream import ic
from plotnine import *

# Local imports
from algorl.logs import logging
from algorl.src.tool_box import create_directory
import sys


class CompareAllBanditsAlgos(object):
    """
    Class for the testing of all MAB algorithms
    """
    def __init__(
        self, time_steps:int=100, arms:int=5, number_of_trials:int=5, images_dir:str='images', 
        q_mean:List[float] = None, q_sd:List[float] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Running TestAll MAB")
        self.df_return = pd.DataFrame()
        self.df_action = pd.DataFrame()
        self.time_steps = time_steps
        self.number_of_trials = number_of_trials
        self.arms = arms
        self.images_dir = images_dir
        create_directory(directory_path = self.images_dir)
        self.q_mean = np.random.randn(self.arms) if q_mean is None else q_mean
        self.q_sd = [1] * self.arms if q_sd is None else q_sd
    
    def test_algo(self, algo, col_name:str=None, epsilon:float=.1, UCB_param:float=0.1, initial:float=.0):
        self.epsilon = epsilon
        self.UCB_param = UCB_param
        if col_name is None:
            col_name = algo.__name__
        self.logger.info(f"Running {col_name}")
        rewards, best_actions = [], []
        for _ in range(self.number_of_trials):
            self.logger.info(f"\ttest: {_}")
            #1 New bandits
            bandits = Bandits(number_of_arms = self.arms, initial=initial, q_mean=self.q_mean, q_sd=self.q_sd)
            #2 Simulate
            explore = algo(bandits)
            reward, best_action = explore.simulate(time = self.time_steps)
            rewards.append(reward)
            best_actions.append(best_action)
            bandits.reset_bandit_df()
        # bandits.plot_bandits()
        # bandits.plot_true_mean_vs_estimation(pic_name = f"{col_name}_true_mean_vs_estimation")
        self.df_return[f"{col_name}"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(self.time_steps))
        self.df_action[f"{col_name}"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(self.time_steps))
    
    def return_dfs(self):
        return self.df_return, self.df_action

    def _comparing_plots(self, arg0, arg1, pic_name):
        '''Line charts in ggplot/plotnine'''
        self.logger.info(f"Plotting {pic_name}")
        # arg0.plot.line(figsize=(10, 5))
        # plt.xlabel("Steps")
        # plt.ylabel(arg1)
        # plt.title(f"{pic_name}")
        # plt.legend(bbox_to_anchor=(1., .75))
        # plt.tight_layout()
        # plt.savefig(Path(self.images_dir, f'{pic_name}_old.png'), dpi=300)
        # plt.close()

        arg0['time'] = range(self.time_steps)
        df = pd.melt(arg0, value_vars=arg0.columns[:-1],  id_vars='time', value_name='value')
        g = (
            ggplot(df, aes(x='time', y='value', color='variable', group='variable'))
        ) + geom_line(
        ) + labs(x='Steps', y=arg1, color='Algorithms'#) + theme_classic(
        ) + ggtitle(f"{pic_name}"
        ) + scale_fill_brewer(type="qual", palette = "Pastel1") 
        g.save(Path(self.images_dir, f'{pic_name}.png'), dpi=300)

    def plot_action_taken(self, best_actions, pic_name:str = 'Optimal Actions' ):
        self._comparing_plots(
            best_actions, "% of time the optimal action is taken", pic_name
        )

    def plot_returns(self, tot_return, pic_name:str='Average reward per time step' ):
        self._comparing_plots(tot_return, "Amount", pic_name)


class Bandits():
    '''
    Bandits environment
    '''
    def __init__(
        self, 
        number_of_arms:int = 10,
        q_mean:List[float] = None,
        q_sd:List[float] = None, initial:float=.0,
        bandit_name:List[str]=None, images_dir:str = 'images') -> None:
        self.logger = logging.getLogger(__name__)
        # self.logger.info("Initialize Bandits")

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

    def plot_bandits(self):
        df = pd.DataFrame(
            {name:np.random.normal(mu, sigma, size=1_000) 
            for name, mu, sigma in zip(self.bandit_name, self.q_mean, self.q_sd)})
        df = pd.melt(df, value_vars=self.bandit_name, value_name='value')

        g = (
            ggplot(df, aes(x='variable', y='value', fill='variable'))
        ) + geom_violin(df
        ) + labs(x='Actions', y='Reward distribution', color='Actions'
        ) + ggtitle(f"Distribution of {self.number_of_arms}-bandits"
        ) + geom_sina(alpha=0.1, size=.3
        ) + scale_fill_brewer(type="qual", palette = "Pastel1")
        g.save(Path(self.images_dir, f'{self.number_of_arms}-bandits.png'), dpi=300)

    def return_bandit_df(self) -> pd.DataFrame:
        return self.bandit_df

    def plot_true_mean_vs_estimation(self, pic_name:str='TargetVsEstimation', y_axis:str='q_estimation')-> None:
        '''
        Scatter plot of true mean vs estimation
        '''
        g = (
            ggplot(self.bandit_df.T, aes(x='target', y=y_axis, color=self.bandit_df.columns), 
            )
            + geom_point()
            + labs(x='True Mean', y='Estimated Mean', color='Bandits')
            + ggtitle(f"Target vs estimated values ({self.number_of_arms}-bandits)")
        ) + geom_segment( # Add diagonal line
            aes(x = min(self.q_mean), xend = max(self.q_mean),
                y = min(self.q_mean), yend = max(self.q_mean),
                ), color = 'black', linetype='dashed', alpha=.5
        ) + geom_text(self.bandit_df.T, aes(x='target', y=y_axis, label=self.bandit_df.columns),
            ha='left', nudge_x=0.05, color='black'
        )
        g.save(Path(self.images_dir, f'{pic_name}.png'), dpi=300)


class BernoulliBandits(Bandits):
    def __init__(
        self, number_of_arms: int = 10, q_mean: List[float] = None, q_sd: List[float] = None, 
        initial: float = 1, bandit_name: List[str] = None, images_dir: str = 'images') -> None:
        ''''''
        self.logger = logging.getLogger(__name__)
        # self.logger.info("Initialize Bernoulli Bandits")
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
    '''
    Parent class that contains functions that are used in multiple algorithms
    '''
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _step(self, action) -> float:
        """
        This function returns the reward for the action taken
        """
        assert action in self.bandits.bandit_name, f"{action} is not a valid action"

        reward = np.random.normal(
            self.bandits.bandit_df[action]['target'], 
            self.bandits.bandit_df[action]['true_sd'], size=1)[0]
        self.logger.debug(f"Action {action} reward: {reward}")
        self.bandits.bandit_df[action]['action_count'] += 1

        if self.sample_averages:
            self.logger.debug("Sample averages")
            # Q[action] = Q[action] + (reward - Q[action])/N[action]
            self.bandits.bandit_df[action]['q_estimation'] =\
                self.bandits.bandit_df[action]['q_estimation']+\
                (reward - self.bandits.bandit_df[action]['q_estimation'])/\
                    self.bandits.bandit_df[action]['action_count']
        elif self.step_size is not None:
            self.logger.debug(f"Step size {self.step_size }")
            # Q[action] = Q[action] + step_size*(reward - Q[action])
            self.bandits.bandit_df[action]['q_estimation'] =\
                self.bandits.bandit_df[action]['q_estimation'] +\
                    self.step_size *\
                         (reward - self.bandits.bandit_df[action]['q_estimation'])
        return reward

    def simulate(self, time:int) -> Tuple[List[float], List[float]]:
        """
        This function simulates the action taking process.
        """
        best_action = self.bandits.return_bandit_df().loc['target', :].idxmax()
        best_action_count = 0
        best_action_percentage = []
        for num in range(time):
            self.logger.debug(f"Time: {num}")
            action = self._act(num)
            self.tot_return.append(self._step(action))
            if action == best_action:
                best_action_count += 1
            best_action_percentage.append(best_action_count/(num+1))
        self.best_action_percentage = best_action_percentage
        return self.tot_return, self.best_action_percentage


class OnlyExploration(MABFunctions):
    def __init__(
        self, 
        bandits:Bandits,
        sample_averages:bool=True, 
        step_size:float=None) -> None:
        MABFunctions.__init__(self)
        self.bandits = bandits
        self.tot_return = []
        self.sample_averages = sample_averages
        self.step_size = step_size
        # self.logger.info("Initialize OnlyExploration")

    def _act(self, _:int) -> str:
        """
        This function returns a random action 
        """
        return np.random.choice(self.bandits.bandit_name)


class OnlyExploitation(MABFunctions):
    def __init__(
        self, bandits:Bandits, sample_averages:bool=True, 
        step_size:float=None) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize OnlyExploitation")
        self.bandits = bandits
        self.tot_return = []
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.fix_choice = np.random.choice(self.bandits.bandit_name)

    def _act(self, _:int) -> str:
        """
        This function returns the known a priori best action
        """
        # ic(self.fix_choice)
        return self.fix_choice


class Greedy(MABFunctions):
    """
    This code allows pure, epsilon-greedy with action value or step size with or without optimistic initial values. 
    
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 32.
    A simple bandits algorithm
    Initialize, for a = 1 to k: 
    Q(a) <- 0
    N(a) <- 0
    Loop for⇢ever:
        A <- either: arg max_a Q(a) with probability 1 - e (breaking ties randomly)
            or: a random action with probability e
        R <- bandits(A)
        N(A) N(A)+1
        Q(A) Q(A)+ 1/N(A) * (R - Q(A))
    """
    
    def __init__(
        self, bandits:Bandits, epsilon:float=.1, 
        sample_averages:bool=True, step_size:float=None, initial:float=.0
        ) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize Greedy")
        self.bandits = bandits
        self.epsilon = epsilon
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.tot_return = []
        self.bandits.bandit_df.loc['q_estimation', :] += initial

    def _act(self, _:int) -> str:
        """
        This function returns the action to be taken based on the epsilon greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.bandits.bandit_name)
        return np.random.choice(
                self.bandits.bandit_df.loc['q_estimation', :][self.bandits.bandit_df.loc['q_estimation', :] ==\
                    self.bandits.bandit_df.loc['q_estimation', :].max()].index)


class UCB(MABFunctions):
    """
    Upper Confidence Bound (UCB) algorithm.
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 35
    """
    def __init__(
        self, bandits:Bandits, 
        sample_averages:bool=True, step_size:float=None, UCB_param:float=0.1
        ) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize UCB")
        self.bandits = bandits
        self.UCB_param = UCB_param
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.tot_return = []

    def _act(self, num:int) -> pd.DataFrame: 
        # It does a first exploration of all options before using UCB 
        if num < len(self.bandits.bandit_name):
            return self.bandits.bandit_name[num]
        
        # action = np.argmax(Q + c * np.sqrt(np.log(e)/N))
        UCB_estimation = self.bandits.bandit_df.loc['q_estimation', :] + \
            self.UCB_param * np.sqrt(
                np.log(num + 1) / (self.bandits.bandit_df.loc['action_count', :]))
        
        return self.bandits.bandit_df.columns[np.random.choice(np.where(UCB_estimation == np.max(UCB_estimation))[0])]


class GBA(MABFunctions): 
    """
    Gradient bandits Algorithm
    Reference:
    --------------------
    - Reinforcement Learning: An Introduction. Sutton and Barto. 2nd Edition. Page 37
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 118
    """
    def __init__(
        self, bandits:Bandits, decay_ratio:float=0.04,
        sample_averages:bool=True, step_size:float=None, init_temp=100_000_000, min_temp=0.01 #float('inf')
        ) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize GBA/SoftMax")
        self.bandits = bandits
        self.sample_averages = sample_averages
        self.step_size = step_size
        self.tot_return = []
        self.decay_ratio = decay_ratio
        # self.init_temp = min(init_temp, sys.float_info.max)
        self.init_temp = init_temp
        # self.min_temp = max(min_temp, np.nextafter(np.float32(0), np.float32(1)))
        self.min_temp = min_temp
        self.logger.debug(f'Lin SoftMax {init_temp}, {min_temp}, {decay_ratio}')

    def _act(self, num:int) -> str:
        """
        This function returns the action to be taken based on the epsilon greedy policy.
        """
        decay_episodes = num+1 * self.decay_ratio
        temp = 1 - np.exp(1) / decay_episodes

        temp *= (self.init_temp - self.min_temp)
        temp += self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)

        Q = np.random.normal(self.bandits.bandit_df.loc['q_estimation', :], self.bandits.bandit_df.loc['estimated_sd', :])
        scaled_Q = Q / temp
        norm_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(norm_Q)
        probs = exp_Q / np.sum(exp_Q)

        return np.random.choice(self.bandits.bandit_name, p=probs)


class BernoulliThompsonSampling(MABFunctions):
    """
    The Thompson sampling strategy is a sample- based probability matching strategy that allows us to use 
    
    Bayesian techniques to balance the exploration and exploitation trade-off.
    A simple way to implement this strategy is to keep track of each Q-value as a Gaussian distribution. 
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

    Reference:
    --------------------
    - Grokking Deep Reinforcement Learning by Miguel Morales. Page 122, Grokking.
    - A tutorial on Thompson sampling. Page 15
    """
    def __init__(
        self, bandits:Bandits, alpha = 1, beta = 1, bandit_type:str='BernTS',
        ) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize BernoulliThompsonSampling")
        self.bandits = bandits
        self.alpha = alpha
        self.beta = beta
        self.bandit_type = bandit_type
        self.tot_return = []

    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        self.logger.debug(action)
        # Compute Bernoulli distribution
        reward = np.random.binomial(1, self.bandits.bandit_df[action]['target'], size=1)[0]
        self.logger.debug(reward)

        self.bandits.bandit_df[action]['action_count'] += 1
        self.bandits.bandit_df[action]['alpha'] += reward
        self.bandits.bandit_df[action]['beta'] += 1-reward

        assert np.isclose(np.sum(self.bandits.bandit_df.loc['target', :]), 1.0), \
        f"The sum of all probabilities is not 1.0 ({np.sum(self.bandits.bandit_df.loc['target', :])})"
        return reward

    def _act(self, _:int) -> str:        
        if self.bandit_type == 'BernTS':
            # Compute Bernoulli distributions
            self.bandits.bandit_df.loc['theta_hat', :] = \
                np.random.beta(a=self.bandits.bandit_df.loc['alpha', :], b=self.bandits.bandit_df.loc['beta', :])
            self.logger.debug(self.bandits.bandit_df.loc['theta_hat', :])

        elif self.bandit_type == 'BernGreedy':
            # Compute Bernoulli distributions
            self.bandits.bandit_df.loc['theta_hat', :] = \
                self.bandits.bandit_df.loc['alpha', :]/(self.bandits.bandit_df.loc['alpha', :]+self.bandits.bandit_df.loc['beta', :])
            self.logger.debug(self.bandits.bandit_df.loc['theta_hat', :])
        else:
            raise ValueError(f'bandits type {self.bandit_type} not supported')

        # select action
        return np.random.choice(
            self.bandits.bandit_df.loc['theta_hat', :][self.bandits.bandit_df.loc['theta_hat', :] ==\
                self.bandits.bandit_df.loc['theta_hat', :].max()].index)


class GaussianThompsonSampling(MABFunctions):
    '''
    Reference:
    --------------------
    https://en.wikipedia.org/wiki/Conjugate_prior
    Normal with known variance σ2

    Conjugate Bayesian analysis of the Gaussian distribution
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

    Example 4.1, pag 21
    A tutorial on Thompson sampling, Russo 
    '''
    def __init__(
        self, bandits:Bandits, q_estimation:float=0, estimated_sd:float=100) -> None:
        MABFunctions.__init__(self)
        # self.logger.info("Initialize GaussianThompsonSampling")
        self.bandits = bandits
        self.tot_return = []
        self.bandits.bandit_df.loc['theta_hat', :] = 0
        self.bandits.bandit_df.loc['reward', :] = 0
        self.estimated_sd = [estimated_sd]*self.bandits.bandit_df.shape[1] # prior_sigma
        self.bandits.bandit_df.loc['estimated_sd', :] = self.estimated_sd  # post_sigma
        self.q_estimation = [q_estimation]*self.bandits.bandit_df.shape[1] # prior_sigma
        self.bandits.bandit_df.loc['q_estimation', :] = self.q_estimation  # post_sigma


    def _step(self, action) -> None:
        """
        This function updates the action value estimates.
        """
        self.logger.debug(action)
        
        # Compute Bernoulli distribution
        reward = np.random.normal(
            self.bandits.bandit_df[action]['target'], 
            self.bandits.bandit_df[action]['true_sd'], size=1)[0]
        self.logger.debug(reward)

        self.bandits.bandit_df[action]['reward'] += reward
        self.bandits.bandit_df[action]['action_count'] += 1
        
        # Normalwith known variance σ**2
        self.bandits.bandit_df.loc['estimated_sd', :] =\
             np.sqrt((
                 1 / np.array(self.estimated_sd)**2 +\
                 self.bandits.bandit_df.loc['action_count', :] / self.bandits.bandit_df.loc['true_sd', :]**2)**-1)
       
        self.bandits.bandit_df.loc['q_estimation', :] =\
             (self.bandits.bandit_df.loc['estimated_sd', :]**2)*((np.array(self.q_estimation)/ np.array(self.estimated_sd)**2) +\
                 (self.bandits.bandit_df.loc['reward', :]/self.bandits.bandit_df.loc['true_sd', :]**2))
        return reward

    def _act(self, _:int) -> str:        
        # Compute value from estimated distribution 
        self.bandits.bandit_df.loc['theta_hat', :] = \
            np.random.normal(self.bandits.bandit_df.loc['q_estimation', :], self.bandits.bandit_df.loc['estimated_sd', :])
        self.logger.debug(self.bandits.bandit_df.loc['theta_hat', :])

        # select action
        return( np.random.choice(
            self.bandits.bandit_df.loc['action_count', :][self.bandits.bandit_df.loc['theta_hat', :] ==\
                self.bandits.bandit_df.loc['theta_hat', :].max()].index) )