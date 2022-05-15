"""Simple script to run snips of code"""
# Standard Libraries
import os
import sys
os.chdir(os.path.dirname(__file__))

# Third party libraries
from icecream import ic

# Local imports
from algorl.logs import logging
from algorl.src.bandit import *

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
class MABExamples(ABC):
    '''Operations'''
    @abstractmethod
    def mab():
        pass

class GreedySampleAverages(MABExamples):
    '''
    '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits)
        greedy.simulate(time = times)
        
        bandits.plot_true_mean_vs_estimation(pic_name)
        print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())


class GreedyStepSize(MABExamples):
    '''
    Test the greedy algorithm with sample_averages
    '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits, sample_averages=False, step_size=0.1)
        greedy.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())


class greedy_sample_averages_with_initials(MABExamples):
    '''
    Test the greedy algorithm with sample_averages
    '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits, )
        greedy.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())


class UCBRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        ucb = UCB(bandits)
        ucb.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        print((ucb.bandits.bandit_df.loc['action_count', :]*ucb.bandits.bandit_df.loc['q_estimation', :]).sum())


class GaussianThompsonSamplingRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        gts = GaussianThompsonSampling(bandits=bandits)
        gts.simulate(time=times)    
        bandits.plot_true_mean_vs_estimation(pic_name)
        print(gts.bandits.bandit_df)
        print((gts.bandits.bandit_df.loc['action_count', :]*gts.bandits.bandit_df.loc['q_estimation', :]).sum())


class GBARun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        gba = GBA(bandits)
        gba.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        print((gba.bandits.bandit_df.loc['action_count', :]*gba.bandits.bandit_df.loc['q_estimation', :]).sum())

"""
class BernoulliThompsonSamplingRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        bernoulli_bandits = BernoulliBandits(number_of_arms = 5, q_mean=[0.1, 0.2, 0.5, 0.05, 0.15])
        # BernTS
        ts = BernoulliThompsonSampling(bandits=bernoulli_bandits, bandit_type = "BernTS")
        BernTS_return, BernTS_actions = ts.simulate(time=times)
        bernoulli_bandits.plot_true_mean_vs_estimation(pic_name=pic_name, y_axis = 'theta_hat')

        # BernGreedy
        ts = BernoulliThompsonSampling(bandits=bernoulli_bandits, bandit_type = "BernGreedy")
        BernGreedy_return, BernGreedy_actions = ts.simulate(time=times)

        CompareAllBanditsAlgos(time_steps=times).plot_returns(pd.DataFrame({
            'BernTS':np.cumsum(BernTS_return), 'BernGreedy':np.cumsum(BernGreedy_return)}) )
        CompareAllBanditsAlgos(time_steps=times).plot_action_taken(pd.DataFrame({ 
            'BernTS':BernTS_actions, 'BernGreedy':BernGreedy_actions}))
# """

def main(arms=5, number_of_trials=5):
    """Runs the main script"""
    logger.info("Starting CompareAllBanditsAlgos MAB")
    test_all = CompareAllBanditsAlgos(arms=arms, number_of_trials=number_of_trials #q_mean=[1,2,3,4,5]
    )
    test_all.test_algo(OnlyExploration)
    test_all.test_algo(OnlyExploitation)
    test_all.test_algo(GaussianThompsonSampling)
    test_all.test_algo(GBA)
    for epsilon in [.1, .5, .9]:
        test_all.test_algo(Greedy, epsilon=epsilon, col_name = f"Greedy {epsilon}")
        test_all.test_algo(Greedy, epsilon=epsilon, col_name = f"Optimistic {epsilon}", initial=5)
        test_all.test_algo(UCB, UCB_param = epsilon, col_name = f"UCB {epsilon}")
    tot_return, best_actions = test_all.return_dfs()
    test_all.plot_returns(tot_return)
    test_all.plot_action_taken(best_actions)

if __name__ == "__main__":
    bandits = Bandits(number_of_arms = 5)
    bandits.plot_bandits()
    for mab_examples in MABExamples.__subclasses__():
        mab_examples().mab(bandits, pic_name=f"{mab_examples.__name__}", times=500)
    # main(arms=5, number_of_trials=50)