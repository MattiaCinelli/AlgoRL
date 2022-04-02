"""Simple script to run snips of code"""
# Standard Libraries
import sys

# Third party libraries
from icecream import ic

# Local imports
from algorl.logs import logging
from algorl.src.bandit import *

logger = logging.getLogger(__name__)

def GreedyAlgo():
    bandits = Bandits(number_of_arms = 5) 
    greedy = Greedy(bandits)
    greedy.simulate(time = 500)
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

    TestAll().plot_returns(pd.DataFrame({
        'BernTS':np.cumsum(BernTS_return), 'BernGreedy':np.cumsum(BernGreedy_return)}))
    TestAll().plot_action_taken(pd.DataFrame({ 
        'BernTS':BernTS_actions, 'BernGreedy':BernGreedy_actions}))
    
def GaussianThompsonSampling_test():
    bandits = Bandits(number_of_arms = 5)
    gts = GaussianThompsonSampling(bandits=bandits)
    gts.simulate(time=500)    
    bandits.plot_true_mean_vs_estimation()
    print(gts.bandits.bandit_df)

def main(arms=5, number_of_trials=5):
    """Runs the main script"""
    logger.info("Starting TestAll MAB")
    test_all = TestAll(arms=arms, number_of_trials=number_of_trials #q_mean=[1,2,3,4,5]
    )
    test_all.test_algo(OnlyExploration)
    test_all.test_algo(OnlyExploitation)
    test_all.test_algo(GaussianThompsonSampling)
    for epsilon in [.1, .5, .9]:
        test_all.test_algo(Greedy, f"Greedy {epsilon}")
        test_all.test_algo(UCB, f"UCB {epsilon}")
    tot_return, best_actions = test_all.return_dfs()
    test_all.plot_returns(tot_return)
    test_all.plot_action_taken(best_actions)

if __name__ == "__main__":
    main(arms=5, number_of_trials=100)
    # BernoulliThompsonSampling_test()
    # GaussianThompsonSampling_test()