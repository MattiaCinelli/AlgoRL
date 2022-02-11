"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np

# Third party libraries
from icecream import ic

# Local imports
from algorl.logs import logging
from algorl.src.bandit import Bandits, Greedy, UCB


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


if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main MAB.py")

    bandit = Bandits(
        number_of_arms=10, 
        q_mean=np.random.randn(10),  # q_mean=[1,2,3,4,5,6,7,8,9,10] # 
        # initial=4
        )
    bandit.plot_bandits()
    
    # ic(bandit.return_bandit_df())

    test_greedy(bandit)
    test_UCB(bandit)
