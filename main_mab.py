"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np
import sys
# Third party libraries
from icecream import ic
import pandas as pd
# Local imports
from algorl.logs import logging
from algorl.src.bandit import Bandits, OnlyExploration, OnlyExploitation, Greedy, UCB, GBA


def test_greedy(bandit):
    greedy = Greedy(bandit, epsilon=0.1) # sample_averages=False, step_size=.999
    greedy.simulate(time=1_000)
    greedy.plot_action_taken()
    bandit.reset_bandit_df()

def test_UCB(bandit):
    ucb = UCB(bandit)
    ucb.simulate(time=1_000)
    ucb.plot_action_taken(pic_name='UCB')
    bandit.reset_bandit_df()

def test_GBA(bandit):
    gba = GBA(bandit)
    gba.simulate(time=1_000)
    gba.plot_action_taken()
    bandit.reset_bandit_df()

def test_all():
    df_return = pd.DataFrame()
    df_action = pd.DataFrame()
    time_steps = 100
    number_of_trials = 5
    arms = 5

    # Only Exploration 
    logger.info("Only Exploration")
    rewards, best_actions = [], []
    for _ in range(number_of_trials):
        #1 New bandits
        bandit = Bandits(number_of_arms = arms)
        #2 Simulate
        explore = OnlyExploration(bandit)
        reward, best_action =  explore.simulate(time = time_steps)
        rewards.append(np.cumsum(reward))
        best_actions.append(best_action)
        bandit.reset_bandit_df()
    df_return["Exploration"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(time_steps))
    df_action["Exploration"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(time_steps))

    ## Only Exploitation
    logger.info("Only Exploitation")
    rewards, best_actions = [], []
    for _ in range(number_of_trials):
        #1 New bandits
        bandit = Bandits(number_of_arms = arms)
        #2 Simulate
        explore = OnlyExploitation(bandit)
        reward, best_action =  explore.simulate(time = time_steps)
        rewards.append(np.cumsum(reward))
        best_actions.append(best_action)
        bandit.reset_bandit_df()
    df_return["Exploitation"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(time_steps))
    df_action["Exploitation"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(time_steps))

    # '''
    # for epsilon in [.1, .3, .5, .7, .9]:
    for epsilon in [.1, .5, .9]:
        # Greedy
        logger.info(f"For Greedy, epsilon: {epsilon}")
        rewards, best_actions = [], []
        for _ in range(number_of_trials):
            logger.info("\ttest: {}".format(_))
            #1 New bandits
            bandit = Bandits(number_of_arms = arms)
            greedy = Greedy(bandit=bandit, epsilon = epsilon)
            #2 Simulate
            reward, best_action = greedy.simulate(time = time_steps)
            rewards.append(np.cumsum(reward))
            best_actions.append(best_action)
            bandit.reset_bandit_df()
        #3 Update dataframe
        df_return[f"Greedy {epsilon}"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(time_steps))
        df_action[f"Greedy {epsilon}"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(time_steps))

        # UCB
        logger.info(f"For UCB, epsilon: {epsilon}")
        rewards, best_actions = [], []
        for _ in range(number_of_trials):
            logger.info("\ttest: {}".format(_))
            #1 New bandits
            bandit = Bandits(number_of_arms = arms)
            ucb = UCB(bandit, epsilon = epsilon)
            #2 Simulate
            reward, best_action = ucb.simulate(time = time_steps)
            rewards.append(np.cumsum(reward))
            best_actions.append(best_action)
            bandit.reset_bandit_df()
        #3 Update dataframe
        df_return[f"UCB {epsilon}"] = pd.Series(np.mean([rewards], axis=1)[0], index=range(time_steps))
        df_action[f"UCB {epsilon}"] = pd.Series(np.mean([best_actions], axis=1)[0], index=range(time_steps))
    
    #4 Plot all
    ucb.plot_returns(df_return)
    ucb.plot_action_taken(df_action)
    # '''

if __name__ == "__main__":
    logger = logging.getLogger("Main")
    logger.info("Running Main MAB.py")

    test_all()
    # bandit = Bandits(
    #     number_of_arms=10, 
    #     # q_mean=np.random.randn(10), 
    #      q_mean=[1,2,3,4,5,6,7,8,9,10] # 
    #     # initial=4
    #     )
    # bandit.plot_bandits()
    
    # test_greedy(bandit)
    # test_UCB(bandit)
    # test_GBA(bandit)



    # """

