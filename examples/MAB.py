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

class OnlyExploitationRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        best_action_percentages = []
        for _ in range(150):
            oe = OnlyExploitation(bandits)
            tot_return, best_action_percentage = oe.simulate(time = times)
            # bandits.plot_true_mean_vs_estimation(pic_name)
            # regret = oe.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return)
            best_action_percentages.append(best_action_percentage)
        # print(best_action_percentage)
        print(np.mean([best_action_percentages], axis=1)[0])
# """
class GreedySampleAverages(MABExamples):
    ''' Test the greedy algorithm with sample_averages '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits)
        tot_return, best_action_percentage = greedy.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        regret = greedy.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return)
        ic(regret)
        # print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())


class GreedyStepSize(MABExamples):
    ''' Test the greedy algorithm with step_size '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits, sample_averages=False, step_size=0.1)
        tot_return, best_action_percentage = greedy.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        # print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())
        ic(greedy.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return))


class GreedySampleAveragesWithInitials(MABExamples):
    '''
    Test the greedy algorithm with optimistic initial values
    '''
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        greedy = Greedy(bandits, sample_averages=True, initial=10)
        tot_return, best_action_percentage = greedy.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        # print((greedy.bandits.bandit_df.loc['action_count', :]*greedy.bandits.bandit_df.loc['q_estimation', :]).sum())
        ic(greedy.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return))


class UCBRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        ucb = UCB(bandits)
        tot_return, best_action_percentage = ucb.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        # print((ucb.bandits.bandit_df.loc['action_count', :]*ucb.bandits.bandit_df.loc['q_estimation', :]).sum())
        ic(ucb.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return))


class GaussianThompsonSamplingRun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        gts = GaussianThompsonSampling(bandits=bandits)
        tot_return, best_action_percentage = gts.simulate(time=times)    
        bandits.plot_true_mean_vs_estimation(pic_name)
        # print((gts.bandits.bandit_df.loc['action_count', :]*gts.bandits.bandit_df.loc['q_estimation', :]).sum())
        ic(gts.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return))


class GBARun(MABExamples):
    def __init__(self):
        super().__init__()

    def mab(self, bandits, pic_name, times):
        gba = GBA(bandits)
        tot_return, best_action_percentage = gba.simulate(time = times)
        bandits.plot_true_mean_vs_estimation(pic_name)
        # print((gba.bandits.bandit_df.loc['action_count', :]*gba.bandits.bandit_df.loc['q_estimation', :]).sum())
        ic(gba.bandits.bandit_df.loc['target', :].max()*times - np.sum(tot_return))
# """
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

def main(arms=5, number_of_trials=5, time_steps=None, q_mean=None, q_sd=None, initial=0, images_dir='images'):
    """Runs the main script"""
    logger.info("Starting CompareAllBanditsAlgos MAB")
    test_all = CompareAllBanditsAlgos(
        arms=arms, number_of_trials=number_of_trials,
        time_steps=time_steps, 
        q_mean=q_mean, q_sd=q_sd, images_dir=images_dir)
    
    test_all.test_algo(OnlyExploration)
    test_all.test_algo(OnlyExploitation)
    # test_all.test_algo(GaussianThompsonSampling)
    test_all.test_algo(GBA)
    for epsilon in [.9]:
        test_all.test_algo(Greedy, epsilon=epsilon, col_name = f"Greedy \u03B5 {epsilon}")
        test_all.test_algo(UCB,  UCB_param=epsilon, col_name = f"UCB \u03B5 {epsilon}")
        if initial>0:
            test_all.test_algo(Greedy, epsilon=epsilon, col_name = f"Optimistic \u03B5 {epsilon}", initial=initial)
    tot_return, best_actions = test_all.return_dfs()
    test_all.plot_returns(tot_return)
    test_all.plot_action_taken(best_actions)


if __name__ == "__main__":
    bandits = Bandits(number_of_arms = 4, q_mean=[1,2,3,4], q_sd=[1.0, 1.0, 1.0, 1.0])
    bandits.plot_bandits()

    # for mab_examples in MABExamples.__subclasses__():
    #     mab_examples().mab(bandits, pic_name=f"{mab_examples.__name__}", times=100)

    # Example 1
    main(arms=4, number_of_trials=250, time_steps=75, q_mean=[1,2,3,4], q_sd=[1.0, 1.0, 1.0, 1.0], initial=5)

    # Example 2
    main(arms=5, number_of_trials=250, time_steps=50, images_dir="images2", initial=3)