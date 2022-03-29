"""Simple script to run snips of code"""
# Standard Libraries
import numpy as np
import sys
# Third party libraries
from icecream import ic
import pandas as pd
# Local imports
from algorl.logs import logging
from algorl.src.bandit import *

logger = logging.getLogger("Main MAB")

if __name__ == "__main__":
    test_all = TestAll()
    test_all.test_algo(OnlyExploration)
    test_all.test_algo(OnlyExploitation)
    for epsilon in [.1, .5, .9]:
        test_all.test_algo(Greedy, f"Greedy {epsilon}")
        test_all.test_algo(UCB, f"UCB {epsilon}")

    test_all.plot_returns(tot_return=test_all.return_dfs()[0])
    test_all.plot_returns(tot_return=test_all.return_dfs()[1])

