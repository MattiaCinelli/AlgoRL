"""Simple script to run snips of code"""
# Standard Libraries
import sys

# Third party libraries
from icecream import ic

# Local imports
from algorl.logs import logging
from algorl.src.bandit import *

logger = logging.getLogger(__name__)

def greedy_sample_averages():
    '''
    Test the greedy algorithm with sample_averages
    '''
    bandits = Bandits(number_of_arms = 5) 
    greedy = Greedy(bandits)
    greedy.simulate(time = 500)
    bandits.plot_true_mean_vs_estimation()

def greedy_step_size():
    '''
    Test the greedy algorithm with sample_averages
    '''
    bandits = Bandits(number_of_arms = 5) 
    greedy = Greedy(bandits, sample_averages=False, step_size=0.1)
    greedy.simulate(time = 500)
    bandits.plot_true_mean_vs_estimation()

def greedy_sample_averages_with_initials():
    '''
    Test the greedy algorithm with sample_averages
    '''
    bandits = Bandits(number_of_arms = 5, initial=5) 
    greedy = Greedy(bandits, )
    greedy.simulate(time = 500)
    bandits.plot_true_mean_vs_estimation()

def UCB_test():
    bandits = Bandits(number_of_arms = 5) 
    ucb = UCB(bandits)
    ucb.simulate(time = 500)
    bandits.plot_true_mean_vs_estimation()

def BernoulliThompsonSampling_test():
    bernoulli_bandits = BernoulliBandits(number_of_arms = 5)#, q_mean=[0.4, 0.6, 0.7, 0.8, 0.9])
    # BernTS
    ts = BernoulliThompsonSampling(bandits=bernoulli_bandits, bandit_type = "BernTS")
    BernTS_return, BernTS_actions = ts.simulate(time=100)
    bernoulli_bandits.plot_true_mean_vs_estimation(y_axis = 'theta_hat')

    # BernGreedy
    ts = BernoulliThompsonSampling(bandits=bernoulli_bandits, bandit_type = "BernGreedy")
    BernGreedy_return, BernGreedy_actions = ts.simulate(time=100)

    CompareAllBanditsAlgos().plot_returns(pd.DataFrame({
        'BernTS':np.cumsum(BernTS_return), 'BernGreedy':np.cumsum(BernGreedy_return)}))
    CompareAllBanditsAlgos().plot_action_taken(pd.DataFrame({ 
        'BernTS':BernTS_actions, 'BernGreedy':BernGreedy_actions}))
    
def GaussianThompsonSampling_test():
    bandits = Bandits(number_of_arms = 5)
    gts = GaussianThompsonSampling(bandits=bandits)
    gts.simulate(time=500)    
    bandits.plot_true_mean_vs_estimation()
    print(gts.bandits.bandit_df)

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
        test_all.test_algo(UCB, UCB_param = epsilon, col_name = f"UCB {epsilon}")
    tot_return, best_actions = test_all.return_dfs()
    test_all.plot_returns(tot_return)
    test_all.plot_action_taken(best_actions)

def GBA_test():
    bandits = Bandits(number_of_arms = 5) 
    gba = GBA(bandits)
    gba.simulate(time = 500)
    bandits.plot_true_mean_vs_estimation()

if __name__ == "__main__":
    # greedy_sample_averages_test()
    # greedy_step_size_test()
    # greedy_sample_averages_test_with_initials()
    # UCB_test()
    # BernoulliThompsonSampling_test()
    # GaussianThompsonSampling_test()
    # GBA_test()
    main(arms=5, number_of_trials=50)
