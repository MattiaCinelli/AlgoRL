"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np
import sys
# Third party libraries
from icecream import ic
import pandas as pd
# Local imports
from algorl.logs import logging
from algorl.src.bandit import Bandits, Greedy, UCB, GBA


def test_greedy(bandit):
    greedy = Greedy(bandit, epsilon=0.1, # sample_averages=False, step_size=.999
)
    greedy.simulate(time=1_000)
    greedy.plot_action_taking()
    bandit.reset_bandit_df()

def test_UCB(bandit):
    ucb = UCB(bandit)
    ucb.simulate(time=1_000)
    ucb.plot_action_taking()
    bandit.reset_bandit_df()

def test_GBA(bandit):
    gba = GBA(bandit)
    gba.simulate(time=1_000)
    gba.plot_action_taking()
    bandit.reset_bandit_df()

if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main MAB.py")

    bandit = Bandits(
        number_of_arms=10, 
        # q_mean=np.random.randn(10), 
         q_mean=[1,2,3,4,5,6,7,8,9,10] # 
        # initial=4
        )
    bandit.plot_bandits()
    
    # test_greedy(bandit)
    # test_UCB(bandit)
    # test_GBA(bandit)
    
    # greedy = Greedy(bandit, epsilon=0.1)
    # rewards = greedy.simulate(time=1_000)
    # greedy.plot_action_taking()
    # print(len(rewards))
    # print(bandit.return_bandit_df())
    # greedy.plot_returns(rewards)
    # print(np.cumsum(rewards))
    """
    df = pd.DataFrame()
    for epsilon in [0.2, 0.4, 0.6, 0.8, 1.0]:
        logger.info("For Greedy, epsilon: {}".format(epsilon))
        rewards = []
        time_steps = 500
        for _ in range(10):
            logger.info("\ttest: {}".format(_))
            bandit = Bandits(number_of_arms=5)
            greedy = Greedy(bandit, epsilon=epsilon)
            rewards.append(np.cumsum(greedy.simulate(time=time_steps)))
        df[f"Greedy{epsilon}"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(time_steps))
    greedy.plot_returns(df)
    """
